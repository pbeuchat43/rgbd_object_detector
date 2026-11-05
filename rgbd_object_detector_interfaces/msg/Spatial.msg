# The index is local to CaptureDetections.detections.detections; never a tracking ID.
uint32 detection_index
bool valid
string invalid_reason
# Metres in the result's optical frame; all NaN when invalid.
geometry_msgs/Point position
string method
float64 pixel_u
float64 pixel_v
uint32 roi_pixels
uint32 valid_pixels
float64 valid_fraction
float64 depth_mad_m
