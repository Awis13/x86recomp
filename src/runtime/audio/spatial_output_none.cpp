// SpatialOutput where the host has none (spatial_output.h): the mixer
// stays stereo. iOS, iPadOS and tvOS are here too: there the system
// already spatialises an app's stereo for headphones, and a renderer of
// our own would need that turned off for the app; not done yet.

#include "runtime/audio/spatial_output.h"

namespace recomp::audio {

std::unique_ptr<SpatialOutput> OpenSpatialOutput() { return nullptr; }

}  // namespace recomp::audio
