#pragma once
#include <string>

namespace cosmico {

class SnapshotRecorder;
class SnapshotPlayer;

class TimelineUI {
public:
    struct State {
        // Recording
        bool wantRecord = false;
        bool wantStopRecord = false;
        int  captureStride = 10;

        // Playback
        bool wantOpen = false;
        bool wantClose = false;
        std::string filePath;
        bool playing = false;
        float playbackSpeed = 1.0f;
        int  currentFrame = 0;
        bool stepFwd = false;
        bool stepBack = false;
    };

    void render(const SnapshotRecorder* rec, const SnapshotPlayer* player,
                State& state, float dt);
};

} // namespace cosmico
