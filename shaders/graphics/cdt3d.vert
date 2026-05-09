#version 450

struct Vertex {
    float x, y, z, nx, ny, nz, r, g, b;
};

layout(std430, set = 0, binding = 0) readonly buffer VertexBuffer {
    Vertex verts[];
};

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec4 lightDir;    // xyz=direction, w=intensity
    float meshScale;
    int showWireframe;
};

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outBary;

void main() {
    Vertex v = verts[gl_VertexIndex];
    gl_Position = viewProj * vec4(v.x, v.y, v.z, 1.0);

    outColor = vec3(v.r, v.g, v.b);
    outNormal = vec3(v.nx, v.ny, v.nz);

    // Barycentric coordinates for wireframe rendering
    int idx = gl_VertexIndex % 3;
    outBary = vec3(float(idx == 0), float(idx == 1), float(idx == 2));
}
