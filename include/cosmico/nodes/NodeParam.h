#pragma once
#include <string>
#include <variant>
#include <optional>
#include <vector>

namespace cosmico {

using ParamValue = std::variant<float, int, bool, std::string>;

struct ParamDescriptor {
    std::string key;            // "damping", "G", "dt"
    std::string label;          // "Damping Factor"
    std::string tooltip;        // Hover text
    ParamValue  defaultValue;   // Type-establishing default
    std::optional<float> minVal, maxVal;  // Numeric range
    std::vector<std::string> enumLabels;  // Non-empty -> render as dropdown
    enum class UIHint { Default, Slider, Drag, Input, Checkbox, Dropdown };
    UIHint uiHint = UIHint::Default;
    bool inlineDisplay = false; // Show on node body vs. side panel only
    std::string format;         // "%.4f", "%d"
};

struct NodeParam {
    ParamDescriptor descriptor;
    ParamValue      value;
    bool            overridden = false; // true = user set it; false = use global
};

} // namespace cosmico
