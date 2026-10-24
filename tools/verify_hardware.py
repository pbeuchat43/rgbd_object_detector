#!/usr/bin/env python3
"""Explicit isolated hardware acceptance; detailed reports belong in a private archive."""
import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from composition_interfaces.srv import LoadNode, UnloadNode
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rgbd_camera_interfaces.msg import Frame, Status as CameraStatus
from rgbd_object_detector_interfaces.msg import Batch, Status

MODEL_HASH = 'dcf4d911dee975944a3cca895d92803c3181d8dba670759276c4d71901c8040d'
QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    location = (len(values) - 1) * fraction
    lower = int(location)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (location - lower)


class Process:
    def __init__(self, argv, log_path):
        self.log = Path(log_path).open('w')
        self.process = subprocess.Popen([str(value) for value in argv], stdout=self.log, stderr=subprocess.STDOUT)

    @property
    def pid(self):
        return self.process.pid

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=25)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f'Graceful stop timed out; inspect task process PID {self.pid}') from error
        self.log.close()
        check(self.process.returncode in (0, -signal.SIGINT), f'task process {self.pid} exited {self.process.returncode}')


class ResourceSampler:
    def __init__(self, pids):
        self.pids = pids
        self.rows = []
        self.tick_hz = os.sysconf('SC_CLK_TCK')
        self.page_size = os.sysconf('SC_PAGE_SIZE')

    def sample(self):
        ticks = rss = 0
        for pid in self.pids:
            text = Path(f'/proc/{pid}/stat').read_text()
            fields = text[text.rfind(')') + 2:].split()
            ticks += int(fields[11]) + int(fields[12])
            rss += int(fields[21]) * self.page_size
        self.rows.append((time.monotonic(), ticks, rss))

    def summary(self):
        first, last = self.rows[0], self.rows[-1]
        return dict(cpu_percent_one_core=100 * (last[1] - first[1]) / self.tick_hz / (last[0] - first[0]),
                    rss_mib_min=min(row[2] for row in self.rows) / 2**20,
                    rss_mib_max=max(row[2] for row in self.rows) / 2**20,
                    sample_count=len(self.rows))


class Observer(Node):
    def __init__(self, namespace, engine_hash, association=False):
        super().__init__('rgbd_detector_acceptance')
        self.namespace = namespace
        self.engine_hash = engine_hash
        self.rows = []
        self.status = self.camera = None
        self.camera_samples = []
        self.last_sequence = {}
        self.frames, self.pending_batches = OrderedDict(), OrderedDict()
        self.associations = 0
        self.associated_valid_positions = 0
        self.camera_sub = self.create_subscription(CameraStatus, namespace + '/camera/status', self.camera_status, QOS)
        self.status_sub = self.create_subscription(Status, namespace + '/detector/status', self.detector_status, QOS)
        self.batch_sub = self.create_subscription(Batch, namespace + '/detector/batches', self.batch, QOS)
        self.frame_sub = self.create_subscription(Frame, namespace + '/camera/frames', self.frame, QOS) if association else None

    def camera_status(self, message):
        check(message.state != 'error', 'camera error: ' + message.detail)
        self.camera = message
        self.camera_samples.append((message.generation, message.frames, message.last_acquisition_ns))

    def detector_status(self, message):
        check(message.state != 'error', 'detector error: ' + message.detail)
        check(message.pending <= 1, 'detector exceeded pending queue bound')
        self.status = message

    def batch(self, message):
        now = time.monotonic_ns()
        capture = message.capture
        check(message.schema_version == 1 and message.model_sha256 == MODEL_HASH and message.engine_sha256 == self.engine_hash,
              'batch model/artifact identity mismatch')
        check(capture.clock_domain == 'linux_monotonic:' + Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'wrong clock')
        check(capture.sequence > self.last_sequence.get(capture.generation, -1), 'duplicate/regressed output sequence')
        self.last_sequence[capture.generation] = capture.sequence
        check(message.completed_ns >= message.started_ns >= capture.acquisition_ns and
              message.source_age_ns == message.completed_ns - capture.acquisition_ns and
              0 <= message.source_age_ns <= message.max_frame_age_ns, 'stale/invalid timing metadata')
        check(message.preprocess_ns >= 0 and message.inference_ns >= 0 and message.postprocess_ns >= 0 and
              message.preprocess_ns + message.inference_ns + message.postprocess_ns == message.completed_ns - message.started_ns,
              'stage timing boundaries inconsistent')
        header = message.detections.header
        check(header.stamp.sec * 10**9 + header.stamp.nanosec == capture.acquisition_ns + capture.unix_offset_ns,
              'acquisition header mapping lost')
        detections = message.detections.detections
        check(len(detections) == len(message.spatial) <= 20, 'output count/association mismatch')
        objects = []
        for index, (detection, spatial) in enumerate(zip(detections, message.spatial)):
            check(spatial.detection_index == index and detection.header == header and not detection.id, 'detection binding/tracking mismatch')
            check(len(detection.results) == 1, 'unexpected hypotheses')
            hypothesis = detection.results[0].hypothesis
            check(hypothesis.class_id in ('yellow_cone', 'blue_cone') and 0 <= hypothesis.score <= 1, 'invalid class/score')
            point = [spatial.position.x, spatial.position.y, spatial.position.z]
            if spatial.valid:
                check(all(math.isfinite(value) for value in point) and point[2] > 0 and not spatial.invalid_reason, 'invalid valid XYZ')
            else:
                check(all(math.isnan(value) for value in point) and spatial.invalid_reason, 'invalid XYZ represented as a real point')
            objects.append(dict(class_id=hypothesis.class_id, score=hypothesis.score, valid=spatial.valid,
                                xyz=point if spatial.valid else None, reason=spatial.invalid_reason,
                                valid_pixels=spatial.valid_pixels, valid_fraction=spatial.valid_fraction))
        self.rows.append(dict(generation=capture.generation, sequence=capture.sequence,
                              acquired_ns=capture.acquisition_ns, received_ns=now,
                              source_age_ms=message.source_age_ns/1e6, observer_age_ms=(now-capture.acquisition_ns)/1e6,
                              preprocess_ms=message.preprocess_ns/1e6, inference_ms=message.inference_ns/1e6,
                              postprocess_ms=message.postprocess_ns/1e6, objects=objects,
                              calibration_id=message.calibration_id, truncated=message.truncated))
        if self.frame_sub:
            key = (capture.generation, capture.sequence)
            self.pending_batches[key] = message
            self.join(key)
            while len(self.pending_batches) > 8:
                self.pending_batches.popitem(last=False)

    def frame(self, message):
        key = (message.capture.generation, message.capture.sequence)
        self.frames[key] = message
        self.join(key)
        while len(self.frames) > 8:
            self.frames.popitem(last=False)

    def join(self, key):
        if key not in self.frames or key not in self.pending_batches:
            return
        frame, batch = self.frames.pop(key), self.pending_batches.pop(key)
        check(frame.capture == batch.capture and frame.calibration_id == batch.calibration_id and
              frame.depth_sequence == batch.depth_sequence and frame.depth_acquisition_ns == batch.depth_acquisition_ns,
              'hardware exact-bundle association mismatch')
        depth = np.frombuffer(frame.depth.data, '<u2').reshape(frame.depth.height, frame.depth.step//2)[:, :frame.depth.width]
        for detection, spatial in zip(batch.detections.detections, batch.spatial):
            box = detection.bbox
            u, v = box.center.position.x, box.center.position.y
            x1, x2 = max(0, math.ceil(u-box.size_x*batch.roi_fraction/2)), min(frame.depth.width, math.ceil(u+box.size_x*batch.roi_fraction/2))
            y1, y2 = max(0, math.ceil(v-box.size_y*batch.roi_fraction/2)), min(frame.depth.height, math.ceil(v+box.size_y*batch.roi_fraction/2))
            raw = depth[y1:y2, x1:x2]
            metres = raw.astype(np.float64) * frame.depth_unit_m
            valid = metres[(raw != frame.invalid_depth_value) & (metres >= batch.min_depth_m) & (metres <= batch.max_depth_m)]
            check(raw.size == spatial.roi_pixels and valid.size == spatial.valid_pixels, 'hardware ROI/depth sample count mismatch')
            if spatial.valid:
                z = float(np.median(valid))
                k = frame.rgb_info.k
                expected = [(u-k[2])*z/k[0], (v-k[5])*z/k[4], z]
                actual = [spatial.position.x, spatial.position.y, spatial.position.z]
                check(np.allclose(expected, actual, rtol=0, atol=1e-9), 'hardware XYZ does not match its exact depth/calibration')
                check(abs(float(np.median(np.abs(valid-z))) - spatial.depth_mad_m) < 1e-9, 'hardware depth MAD mismatch')
                self.associated_valid_positions += 1
        self.associations += 1

    def wait(self, condition, timeout, message, sampler=None):
        deadline, sample_at = time.monotonic()+timeout, 0
        while not condition() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.03)
            if sampler and time.monotonic() >= sample_at:
                sampler.sample(); sample_at = time.monotonic()+.5
        check(condition(), message)

    def duration(self, seconds, sampler=None):
        deadline = time.monotonic() + seconds
        self.wait(lambda: time.monotonic() >= deadline, seconds+1, 'duration timeout', sampler)

    def reset_measurement(self):
        self.rows.clear(); self.camera_samples.clear()
        self.associations = self.associated_valid_positions = 0
        self.frames.clear(); self.pending_batches.clear()

    def summary(self, sampler):
        rows = self.rows
        result = dict(batch_count=len(rows), empty_batches=sum(not row['objects'] for row in rows),
                      valid_positions=sum(obj['valid'] for row in rows for obj in row['objects']),
                      invalid_positions=sum(not obj['valid'] for row in rows for obj in row['objects']),
                      source_generations=sorted({row['generation'] for row in rows}),
                      exact_associations=self.associations, associated_valid_positions=self.associated_valid_positions,
                      resource=sampler.summary(), batches=rows.copy())
        if len(rows) >= 2:
            result['batch_rate_hz'] = (len(rows)-1)*1e9/(rows[-1]['received_ns']-rows[0]['received_ns'])
        for key in ('source_age_ms', 'observer_age_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms'):
            values = [row[key] for row in rows]
            result[key] = dict(p50=percentile(values,.5), p95=percentile(values,.95), maximum=max(values, default=None))
        if len(self.camera_samples) >= 2:
            first, last = self.camera_samples[0], self.camera_samples[-1]
            if first[0] == last[0] and last[2] > first[2]:
                result['camera_published_hz'] = (last[1]-first[1])*1e9/(last[2]-first[2])
        if self.status:
            result['detector'] = {key: getattr(self.status,key) for key in
                                  ('received','rejected','replaced','stale','superseded','generation_changes','processed','failures','pending','processing')}
        return result


def ros_arguments(namespace, name, parameters=(), remaps=()):
    arguments = ['--ros-args', '-r', '__ns:=' + namespace, '-r', '__node:=' + name]
    for key, value in parameters:
        arguments += ['-p', f"{key}:='{value}'" if isinstance(value, str) else f'{key}:={value}']
    for source, destination in remaps:
        arguments += ['-r', source + ':=' + destination]
    return arguments


def service(observer, suffix, request, service_type):
    client = observer.create_client(service_type, observer.namespace + '/container/_container/' + suffix)
    try:
        observer.wait(client.service_is_ready, 10, 'component service not available: ' + suffix)
        future = client.call_async(request)
        observer.wait(future.done, 20, 'component service did not complete: ' + suffix)
        response = future.result()
        check(response.success, 'component service rejected: ' + response.error_message)
        return response
    finally:
        observer.destroy_client(client)


def load(observer, package, plugin, name, parameters, remaps=()):
    request = LoadNode.Request()
    request.package_name, request.plugin_name = package, plugin
    request.node_name, request.node_namespace = name, observer.namespace
    request.parameters = [Parameter(key, value=value).to_parameter_msg() for key, value in parameters]
    request.extra_arguments = [Parameter('use_intra_process_comms', value=True).to_parameter_msg()]
    request.remap_rules = list(remaps)
    return service(observer, 'load_node', request, LoadNode).unique_id


def unload(observer, identifier):
    request = UnloadNode.Request(); request.unique_id = identifier
    service(observer, 'unload_node', request, UnloadNode)


def run_camera_case(case, args, engine_hash):
    directory = args.output / case
    directory.mkdir()
    namespace = '/rgbd_verify_' + case
    observer = Observer(namespace, engine_hash, association=(case == 'association'))
    processes = []
    camera_parameters = [('device_id', args.device_id), ('fps', 10.0)]
    detector_parameters = [('engine_path', str(args.engine))]
    camera = None
    try:
        if case in ('composed', 'association'):
            binary = Path(get_package_prefix('rclcpp_components')) / 'lib/rclcpp_components/component_container_mt'
            container = Process([binary] + ros_arguments(namespace, 'container'), directory / 'container.log')
            processes.append(container)
            camera_id = load(observer, 'rgbd_camera', 'rgbd_camera::CameraNode', 'camera', camera_parameters)
            load(observer, 'rgbd_object_detector', 'rgbd_object_detector::DetectorNode', 'detector', detector_parameters,
                 ['frames:=camera/frames'])
        else:
            binary = Path(get_package_prefix('rgbd_camera')) / 'lib/rgbd_camera/camera_node'
            camera = Process([binary] + ros_arguments(namespace, 'camera', camera_parameters), directory / 'camera.log')
            processes.append(camera)
            binary = (args.test_binaries / 'stress_detector' if case == 'overload' else
                      Path(get_package_prefix('rgbd_object_detector')) / 'lib/rgbd_object_detector/detector_node')
            detector = Process([binary] + ros_arguments(namespace, 'detector', detector_parameters, [('frames','camera/frames')]),
                               directory / 'detector.log')
            processes.append(detector)
        observer.wait(lambda: observer.status is not None and observer.status.backend_ready and
                      observer.camera is not None and observer.camera.state == 'streaming', 40, 'pipeline not ready')
        if case != 'overload':
            observer.wait(lambda: len(observer.rows) >= 10, 20, 'no successful live batches')
        observer.reset_measurement()
        sampler = ResourceSampler([process.pid for process in processes])
        if case == 'overload':
            observer.duration(15, sampler)
            check(observer.status.replaced > 0 and observer.status.received > 60 and
                  observer.status.processed + observer.status.stale > 0, 'overload was not exercised')
        elif case == 'association':
            observer.wait(lambda: observer.associations >= 10, 30, 'insufficient exact hardware associations', sampler)
        else:
            observer.wait(lambda: len(observer.rows) >= args.frames, 40, 'insufficient successful batches', sampler)
        result = observer.summary(sampler)
        result['case'] = case
        result['layout'] = 'composed_intra_process' if case in ('composed','association') else 'separate_processes'
        check(result.get('camera_published_hz', 0) > 8, 'capture did not maintain the requested 10 Hz profile within observation margin')
        if case in ('standalone','composed'):
            if camera:
                camera.stop()
            else:
                unload(observer, camera_id)
            observer.wait(lambda: observer.status.state == 'unavailable' and observer.status.backend_ready and
                          time.monotonic_ns()-observer.status.last_processed_acquisition_ns > observer.status.max_frame_age_ns,
                          4, 'camera loss did not expire detector readiness')
            result['input_loss_expired'] = True
        if case == 'composed':
            before = set(result['source_generations'])
            camera_id = load(observer, 'rgbd_camera', 'rgbd_camera::CameraNode', 'camera', camera_parameters)
            observer.wait(lambda: observer.rows and observer.rows[-1]['generation'] not in before,
                          40, 'camera restart produced no new-generation detections')
            new_generation = observer.rows[-1]['generation']
            observer.reset_measurement()
            restart_sampler = ResourceSampler([processes[0].pid])
            observer.wait(lambda: len(observer.rows) >= 40, 20, 'restart did not sustain processing', restart_sampler)
            check(all(row['generation'] == new_generation for row in observer.rows), 'old result appeared after restart')
            result['after_restart'] = observer.summary(restart_sampler)
            check(observer.status.generation_changes == 1, 'detector did not observe exactly one camera restart')
        if case == 'association':
            result['positive_xyz_observed'] = result['associated_valid_positions'] > 0
        if args.require_positive and case != 'overload':
            check(result['valid_positions'] > 0, 'no positive cone with valid XYZ was observed')
            if case == 'association':
                check(result['positive_xyz_observed'], 'no positive XYZ was checked against its exact source bundle')
        return result
    finally:
        failures = []
        for process in reversed(processes):
            try:
                process.stop()
            except Exception as error:
                failures.append(str(error))
        observer.destroy_node()
        if failures:
            raise RuntimeError('; '.join(failures))


def artifact_checks(args):
    directory = args.output / 'artifacts'
    directory.mkdir()
    original = json.loads(Path(str(args.engine) + '.json').read_text())
    results = []
    for name, field, value in [('wrong_engine_hash','engine_sha256','0'*64),
                               ('wrong_model_hash','model_sha256','0'*64),
                               ('wrong_gpu','compute_major',99)]:
        path = directory / (name + '.engine')
        os.link(args.engine, path)
        metadata = dict(original); metadata[field] = value
        Path(str(path)+'.json').write_text(json.dumps(metadata))
        log = (directory/(name+'.log')).open('w')
        process = subprocess.Popen([str(args.test_binaries/'test_engine'), str(path)], stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=20)
            raise RuntimeError('artifact rejection check timed out')
        finally:
            log.close()
        text = (directory/(name+'.log')).read_text()
        check(code == 1 and 'mismatch' in text, name + ' was not rejected cleanly')
        results.append(dict(case=name, rejected=True, exit_code=code))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id', required=True)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--test-binaries', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--require-positive', action='store_true',
                        help='Require observed cones with valid XYZ; use only with a prepared visible cone scene')
    parser.add_argument('--cases', nargs='+', choices=['artifacts','standalone','composed','overload','association'],
                        default=['artifacts','standalone','composed','overload','association'])
    args = parser.parse_args()
    check(os.environ.get('ROS_DOMAIN_ID') not in (None,'0') and os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE') == 'LOCALHOST',
          'use an isolated nonzero ROS domain with localhost-only discovery')
    check(20 <= args.frames <= 300, '--frames must be 20..300')
    args.engine = args.engine.resolve(); args.test_binaries = args.test_binaries.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    engine_hash = hashlib.sha256(args.engine.read_bytes()).hexdigest()
    report = dict(model_sha256=MODEL_HASH, engine_sha256=engine_hash, ros_domain=os.environ['ROS_DOMAIN_ID'],
                  started_unix=time.time(), cases=[], limitations=[
                      'No surveyed metric accuracy or model recall evidence', 'No physical unplug or endurance test',
                      'No RealSense, driving, browser or deployment integration',
                      'Resource measurements exclude this Python observer; RSS sums are not PSS'])
    rclpy.init()
    try:
        for case in args.cases:
            print('START', case, flush=True)
            result = artifact_checks(args) if case == 'artifacts' else run_camera_case(case,args,engine_hash)
            report['cases'].append(dict(name=case, result=result))
            (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print('PASS', case, flush=True)
        report['passed'] = True
    except Exception as error:
        report['passed'] = False; report['error'] = str(error)
        raise
    finally:
        report['finished_unix'] = time.time()
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        rclpy.shutdown()


if __name__ == '__main__':
    main()
