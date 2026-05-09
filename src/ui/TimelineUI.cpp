#include <cosmico/ui/TimelineUI.h>
#include <cosmico/recording/SnapshotRecorder.h>
#include <cosmico/recording/SnapshotPlayer.h>
#include <imgui.h>
#include <cstdio>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#endif

namespace cosmico {

#ifdef _WIN32
static std::string openFileDialog() {
    char filename[MAX_PATH] = {};
    OPENFILENAMEA ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFilter = "Cosmico Snapshots (*.cosmsnap)\0*.cosmsnap\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameA(&ofn))
        return std::string(filename);
    return {};
}
#else
static std::string openFileDialog() {
    // Fallback: no native dialog on non-Windows
    return {};
}
#endif

static void formatBytes(uint64_t bytes, char* buf, size_t bufSize) {
    if (bytes < 1024)
        snprintf(buf, bufSize, "%llu B", (unsigned long long)bytes);
    else if (bytes < 1024 * 1024)
        snprintf(buf, bufSize, "%.1f KB", bytes / 1024.0);
    else if (bytes < 1024ULL * 1024 * 1024)
        snprintf(buf, bufSize, "%.1f MB", bytes / (1024.0 * 1024.0));
    else
        snprintf(buf, bufSize, "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
}

void TimelineUI::render(const SnapshotRecorder* rec, const SnapshotPlayer* player,
                         State& state, float dt) {
    // Clear one-shot flags
    state.wantRecord = false;
    state.wantStopRecord = false;
    state.wantOpen = false;
    state.wantClose = false;
    state.stepFwd = false;
    state.stepBack = false;

    ImGui::Begin("Timeline###Timeline");

    bool recording = rec && rec->isRecording();
    bool playing_back = player && player->isOpen();

    // ── Recording section ──
    ImGui::SeparatorText("Recording");

    if (!recording && !playing_back) {
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("Capture stride", &state.captureStride);
        state.captureStride = std::max(1, state.captureStride);
        ImGui::SameLine();
        if (ImGui::Button("REC")) {
            state.wantRecord = true;
        }
    } else if (recording) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));
        ImGui::Text("RECORDING");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        if (ImGui::Button("STOP")) {
            state.wantStopRecord = true;
        }
        ImGui::SameLine();
        ImGui::Text("Frames: %u", rec->framesCaptured());
        ImGui::SameLine();
        char sizeBuf[32];
        formatBytes(rec->bytesWritten(), sizeBuf, sizeof(sizeBuf));
        ImGui::Text("Size: %s", sizeBuf);
    } else {
        ImGui::TextDisabled("Close playback to record");
    }

    // ── Playback section ──
    ImGui::SeparatorText("Playback");

    if (!playing_back && !recording) {
        if (ImGui::Button("Open##PlaybackOpen")) {
            std::string path = openFileDialog();
            if (!path.empty()) {
                state.filePath = path;
                state.wantOpen = true;
            }
        }
    } else if (playing_back) {
        // Close button
        if (ImGui::Button("Close##PlaybackClose")) {
            state.wantClose = true;
        }

        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();

        // Transport controls
        if (ImGui::Button("|<")) {
            state.currentFrame = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("<")) {
            state.stepBack = true;
        }
        ImGui::SameLine();
        if (state.playing) {
            if (ImGui::Button("||")) {
                state.playing = false;
            }
        } else {
            if (ImGui::Button(">")) {
                state.playing = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(">|Step")) {
            state.stepFwd = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(">|End")) {
            state.currentFrame = static_cast<int>(player->frameCount()) - 1;
        }

        ImGui::SameLine();
        ImGui::Spacing();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::SliderFloat("Speed", &state.playbackSpeed, 0.1f, 10.0f, "%.1fx");

        // Timeline slider
        int maxFrame = static_cast<int>(player->frameCount()) - 1;
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderInt("##TimelineSlider", &state.currentFrame, 0, maxFrame);

        // Info line
        uint32_t cf = static_cast<uint32_t>(state.currentFrame);
        ImGui::Text("Frame: %u / %u   t = %.3fs   step = %llu",
                     cf, player->frameCount(),
                     player->frameSimTime(cf),
                     (unsigned long long)player->frameSimStep(cf));

        // Advance playback if playing
        if (state.playing) {
            static float accumulator = 0.0f;
            // Base rate: 30 frames per second, scaled by playback speed
            accumulator += dt * state.playbackSpeed * 30.0f;
            int steps = static_cast<int>(accumulator);
            accumulator -= static_cast<float>(steps);
            state.currentFrame += steps;
            if (state.currentFrame >= static_cast<int>(player->frameCount())) {
                state.currentFrame = static_cast<int>(player->frameCount()) - 1;
                state.playing = false;
                accumulator = 0.0f;
            }
        }

        // Single step
        if (state.stepFwd) {
            state.currentFrame = std::min(state.currentFrame + 1, maxFrame);
        }
        if (state.stepBack) {
            state.currentFrame = std::max(state.currentFrame - 1, 0);
        }
    } else {
        ImGui::TextDisabled("Stop recording to play back");
    }

    ImGui::End();
}

} // namespace cosmico
