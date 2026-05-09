#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inBary;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    float meshScale;
    int showWireframe;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = inColor;

    if (showWireframe != 0) {
        // Wireframe via barycentric coordinates
        float minBary = min(inBary.x, min(inBary.y, inBary.z));
        float edgeWidth = fwidth(minBary) * 1.5;
        float edge = smoothstep(0.0, edgeWidth, minBary);

        // Mix: edge color (dark) vs face color
        vec3 edgeColor = color * 0.15;
        color = mix(edgeColor, color, edge);
    }

    outColor = vec4(color, 1.0);
}
