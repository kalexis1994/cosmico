#include <cosmico/nodes/NodeEditorUI.h>
#include <cosmico/nodes/NodeParam.h>

#include <imgui.h>
#include <cstring>
#include <string>
#include <type_traits>
#include <variant>

namespace cosmico {

void NodeEditorUI::renderInlineParam(NodeParam& param, int nodeId) {
    float z = m_editorStack.empty() ? 1.0f : m_editorStack.back().zoomLevel;
    ImGui::PushItemWidth(100.0f * z);
    ImGui::PushID(&param);

    // Override indicator dot
    if (param.overridden) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.2f, 0.8f, 1.0f, 1.0f));
        ImGui::Bullet();
        ImGui::PopStyleColor();
        ImGui::SameLine();
    }

    const auto& d = param.descriptor;
    const char* fmt = d.format.empty() ? nullptr : d.format.c_str();

    // Render based on type
    std::visit([&](auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, float>) {
            float v = val;
            bool changed = false;
            if (d.uiHint == ParamDescriptor::UIHint::Slider && d.minVal && d.maxVal) {
                changed = ImGui::SliderFloat(d.label.c_str(), &v,
                                             *d.minVal, *d.maxVal, fmt ? fmt : "%.3f");
            } else {
                float speed = (d.maxVal && d.minVal) ? (*d.maxVal - *d.minVal) * 0.002f : 0.01f;
                changed = ImGui::DragFloat(d.label.c_str(), &v, speed,
                                           d.minVal.value_or(0.0f),
                                           d.maxVal.value_or(0.0f),
                                           fmt ? fmt : "%.3f");
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int v = val;
            bool changed = false;
            if (!d.enumLabels.empty()) {
                if (ImGui::BeginCombo(d.label.c_str(),
                    (v >= 0 && v < (int)d.enumLabels.size()) ? d.enumLabels[v].c_str() : "?")) {
                    for (int i = 0; i < (int)d.enumLabels.size(); i++) {
                        if (ImGui::Selectable(d.enumLabels[i].c_str(), v == i)) {
                            v = i;
                            changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                changed = ImGui::DragInt(d.label.c_str(), &v, 1.0f,
                                         d.minVal ? (int)*d.minVal : 0,
                                         d.maxVal ? (int)*d.maxVal : 0);
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            bool v = val;
            if (ImGui::Checkbox(d.label.c_str(), &v)) {
                val = v;
                param.overridden = true;
            }
        }
    }, param.value);

    if (!d.tooltip.empty() && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", d.tooltip.c_str());
    }

    ImGui::PopID();
    ImGui::PopItemWidth();
}

void NodeEditorUI::renderParamWidget(NodeParam& param) {
    const auto& d = param.descriptor;
    const char* fmt = d.format.empty() ? nullptr : d.format.c_str();

    ImGui::PushID(&param);

    // Override toggle
    bool ov = param.overridden;
    if (ImGui::Checkbox("##override", &ov)) {
        if (ov) {
            param.overridden = true;
        } else {
            param.value = param.descriptor.defaultValue;
            param.overridden = false;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Override global value");
    }
    ImGui::SameLine();

    if (!param.overridden) ImGui::BeginDisabled();

    ImGui::PushItemWidth(-60.0f);

    std::visit([&](auto& val) {
        using T = std::decay_t<decltype(val)>;
        if constexpr (std::is_same_v<T, float>) {
            float v = val;
            bool changed = false;
            if (d.uiHint == ParamDescriptor::UIHint::Slider && d.minVal && d.maxVal) {
                changed = ImGui::SliderFloat(d.label.c_str(), &v,
                                             *d.minVal, *d.maxVal, fmt ? fmt : "%.3f");
            } else if (d.uiHint == ParamDescriptor::UIHint::Input) {
                changed = ImGui::InputFloat(d.label.c_str(), &v, 0.0f, 0.0f, fmt ? fmt : "%.3f");
            } else {
                float speed = (d.maxVal && d.minVal) ? (*d.maxVal - *d.minVal) * 0.002f : 0.01f;
                changed = ImGui::DragFloat(d.label.c_str(), &v, speed,
                                           d.minVal.value_or(0.0f),
                                           d.maxVal.value_or(0.0f),
                                           fmt ? fmt : "%.3f");
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, int>) {
            int v = val;
            bool changed = false;
            if (!d.enumLabels.empty()) {
                if (ImGui::BeginCombo(d.label.c_str(),
                    (v >= 0 && v < (int)d.enumLabels.size()) ? d.enumLabels[v].c_str() : "?")) {
                    for (int i = 0; i < (int)d.enumLabels.size(); i++) {
                        if (ImGui::Selectable(d.enumLabels[i].c_str(), v == i)) {
                            v = i;
                            changed = true;
                        }
                    }
                    ImGui::EndCombo();
                }
            } else {
                changed = ImGui::DragInt(d.label.c_str(), &v, 1.0f,
                                         d.minVal ? (int)*d.minVal : 0,
                                         d.maxVal ? (int)*d.maxVal : 0);
            }
            if (changed) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, bool>) {
            bool v = val;
            if (ImGui::Checkbox(d.label.c_str(), &v)) {
                val = v;
                param.overridden = true;
            }
        } else if constexpr (std::is_same_v<T, std::string>) {
            char buf[256];
            strncpy(buf, val.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText(d.label.c_str(), buf, sizeof(buf))) {
                val = std::string(buf);
                param.overridden = true;
            }
        }
    }, param.value);

    ImGui::PopItemWidth();

    if (!param.overridden) ImGui::EndDisabled();

    // Reset button
    ImGui::SameLine();
    if (ImGui::SmallButton("R")) {
        param.value = param.descriptor.defaultValue;
        param.overridden = false;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Reset to default");
    }

    ImGui::PopID();
}

} // namespace cosmico
