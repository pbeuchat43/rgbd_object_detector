uint16 schema_version 1
string state
string detail
string source_id
string generation
string clock_domain
string[2] model_ids
int64 observed_ns
int64 expires_ns
int64 last_acquisition_ns
int64 last_render_ns
# Received order: original frames, first model, second model.
uint64[3] received
uint64 matched
uint64 rendered
uint64 rejected
uint64 expired
uint64 evicted
uint64 replaced
uint64 superseded
uint64 no_consumers
uint64 failures
uint32 retained
uint32 capacity
uint32 pending
bool processing
