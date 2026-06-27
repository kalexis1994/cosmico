#version 450

layout(location = 0) in float inSpeed;
layout(location = 1) in float inType;
layout(location = 2) in float inVisualRadius;
layout(location = 3) in vec3 inSphereWorldPos;
layout(location = 4) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    float pointScale;
    float hasPlanetTextures;
    float tanHalfFov;
    float aspectRatio;
    vec4 camRight;    // xyz = right, w = camPos.x
    vec4 camUp;       // xyz = up,    w = camPos.y
    vec4 camForward;  // xyz = fwd,   w = camPos.z
    float simTime;
    float showDarkMatter;
    float lumStrength;  // (unused here; keeps the push block aligned with the struct)
    float coordScale;   // (vertex stage only)
    float exposure;     // camera "exposure": brightness multiplier
};

vec3 speedToColor(float speed) {
    float t = clamp(speed / 20.0, 0.0, 1.0);
    vec3 blue   = vec3(0.2, 0.4, 1.0);
    vec3 yellow = vec3(1.0, 0.9, 0.3);
    vec3 red    = vec3(1.0, 0.2, 0.1);
    if (t < 0.5) return mix(blue, yellow, t * 2.0);
    else         return mix(yellow, red, (t - 0.5) * 2.0);
}

void main() {
    vec2 coord = inUV;
    float r2 = dot(coord, coord);

    if (r2 > 1.0) discard;

    float falloff = 1.0 - sqrt(r2);
    float alpha;
    vec3 color;

    int itype = int(inType + 0.5);
    if (itype == 9) {
        // Asteroid — gray/brown
        alpha = falloff * falloff;
        color = vec3(0.6, 0.55, 0.5);
    } else if (itype == 10) {
        // Kuiper belt object — icy blue-gray
        alpha = falloff * falloff;
        color = vec3(0.5, 0.55, 0.65);
    } else if (inType > 0.0) {
        // Sink / black hole
        alpha = falloff * falloff * 2.0;
        color = vec3(1.0, 1.0, 1.0);
    } else {
        // Luminous particle: Gaussian glow
        alpha = exp(-r2 * 4.0);
        color = speedToColor(inSpeed);
    }

    // Sun lighting only for solar-system minor bodies (asteroids / KBOs)
    if (itype == 9 || itype == 10) {
        vec3 localN = vec3(coord, sqrt(max(1.0 - r2, 0.0)));
        vec3 worldN = normalize(localN.x * camRight.xyz
                              + localN.y * camUp.xyz
                              - localN.z * camForward.xyz);
        vec3 sunDir = normalize(-inSphereWorldPos);
        float diffuse = max(dot(worldN, sunDir), 0.0);
        color *= 0.05 + 0.95 * diffuse;
    }

    // Distance fade: inVisualRadius carries sub-pixel alpha (0..1)
    alpha *= inVisualRadius;
    if (alpha < 0.01) discard;

    // No gl_FragDepth write — uses interpolated depth, enabling early-Z
    // Camera exposure: scale emitted light (premultiplied RGB), keep coverage.
    outColor = vec4(color * alpha * exposure, alpha);
}
