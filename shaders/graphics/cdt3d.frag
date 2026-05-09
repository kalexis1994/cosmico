#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inBary;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 lightDir;    // xyz=direction, w=intensity
    float meshScale;
    int showWireframe;
};

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(inNormal);
    vec3 L = normalize(lightDir.xyz);
    float diff = abs(dot(N, L)) * lightDir.w;
    vec3 color = inColor * (0.2 + diff);  // ambient + diffuse

    if (showWireframe != 0) {
        float minB = min(inBary.x, min(inBary.y, inBary.z));
        float edge = smoothstep(0.0, fwidth(minB) * 1.5, minB);
        color = mix(color * 0.15, color, edge);
    }

    outColor = vec4(color, 1.0);
}
