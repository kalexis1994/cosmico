#version 450

struct Vertex {
    float x, y, r, g, b;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexBuffer {
    Vertex verts[];
};

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    float meshScale;
    int showWireframe;
};

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outBary;

void main() {
    Vertex v = verts[gl_VertexIndex];
    gl_Position = viewProj * vec4(v.x, v.y, 0.0, 1.0);

    outColor = vec3(v.r, v.g, v.b);

    // Barycentric coordinates for wireframe rendering
    int idx = gl_VertexIndex % 3;
    if (idx == 0)      outBary = vec3(1.0, 0.0, 0.0);
    else if (idx == 1) outBary = vec3(0.0, 1.0, 0.0);
    else               outBary = vec3(0.0, 0.0, 1.0);
}
