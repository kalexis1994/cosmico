#version 450

layout(set = 0, binding = 0) uniform sampler2D cmbTex;

layout(push_constant) uniform PC {
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 sphereCenter; // xyz = center, w = radius
    float contrastScale;
    float opacity;
    float reveal;   // 0 = opaque uniform plasma glow, 1 = full CMB pattern
    float showGrid; // >0.5 = overlay the celestial alt-az grid
};

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Ray-sphere intersection
// Returns true if hit, with tNear and tFar
bool intersectSphere(vec3 ro, vec3 rd, vec3 center, float radius, out float tNear, out float tFar) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;

    if (discriminant < 0.0) return false;

    float sqrtD = sqrt(discriminant);
    tNear = -b - sqrtD;
    tFar  = -b + sqrtD;

    return tFar > 0.0;
}

// CMB divergent colormap: blue (cold) -> white (neutral) -> red (hot)
vec3 cmbColormap(float t) {
    // t in [0, 1]: 0 = cold (blue), 0.5 = neutral (white), 1 = hot (red)
    vec3 cold = vec3(0.05, 0.15, 0.65);   // Deep blue
    vec3 cool = vec3(0.35, 0.55, 0.95);   // Light blue
    vec3 neutral = vec3(0.95, 0.95, 0.92); // Off-white
    vec3 warm = vec3(0.95, 0.55, 0.25);   // Orange
    vec3 hot  = vec3(0.70, 0.05, 0.05);   // Deep red

    if (t < 0.25) {
        return mix(cold, cool, t / 0.25);
    } else if (t < 0.5) {
        return mix(cool, neutral, (t - 0.25) / 0.25);
    } else if (t < 0.75) {
        return mix(neutral, warm, (t - 0.5) / 0.25);
    } else {
        return mix(warm, hot, (t - 0.75) / 0.25);
    }
}

void main() {
    // Reconstruct world-space ray from screen UV via inverse view-projection
    vec4 nearNDC = vec4(inUV * 2.0 - 1.0, 0.0, 1.0);
    vec4 farNDC  = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);

    vec4 nearWorld = invViewProj * nearNDC;
    vec4 farWorld  = invViewProj * farNDC;
    nearWorld /= nearWorld.w;
    farWorld  /= farWorld.w;

    vec3 ro = cameraPos.xyz;
    vec3 rd = normalize(farWorld.xyz - nearWorld.xyz);

    vec3 center = sphereCenter.xyz;
    float radius = sphereCenter.w;

    // Ray-sphere intersection
    float tNear, tFar;
    if (!intersectSphere(ro, rd, center, radius, tNear, tFar)) {
        outColor = vec4(0.0);
        return;
    }

    // Choose hit point: if camera is outside sphere, use nearest hit;
    // if inside, use farthest hit (see interior)
    float tHit = (tNear > 0.0) ? tNear : tFar;

    vec3 hitPos = ro + rd * tHit;
    vec3 normal = normalize(hitPos - center);

    // Spherical coordinates for equirectangular UV. Pole on the +Y (world up)
    // axis so the horizon (theta = 90 deg) is the horizontal plane the camera
    // looks along by default, and zenith/nadir are up/down.
    float theta = acos(clamp(normal.y, -1.0, 1.0));           // 0 = zenith (+Y), pi = nadir
    float phi_angle = atan(normal.z, normal.x);                // azimuth around vertical

    vec2 texCoord = vec2(
        (phi_angle + 3.14159265) / (2.0 * 3.14159265),        // [0, 1] azimuthal
        theta / 3.14159265                                      // [0, 1] polar
    );

    // Sample equirectangular CMB map
    float raw = texture(cmbTex, texCoord).r;

    // Apply contrast scaling around 0.5
    float adjusted = (raw - 0.5) * contrastScale + 0.5;
    adjusted = clamp(adjusted, 0.0, 1.0);

    // Map to CMB colormap
    vec3 color = cmbColormap(adjusted);

    // Recombination: before the snapshot the universe is an opaque, near-uniform
    // hot plasma; the anisotropy pattern resolves as 'reveal' goes 0 -> 1.
    vec3 plasma = vec3(0.95, 0.35, 0.12);
    color = mix(plasma, color, reveal);

    // Celestial alt-az grid: parallels of altitude (constant theta) + meridians
    // of azimuth (constant phi), 30 deg spacing, horizon emphasised. Meridian
    // width scaled by 1/sin(theta) so lines stay ~constant width toward the poles.
    if (showGrid > 0.5) {
        const float PI = 3.14159265;
        float s = PI / 6.0;          // 30 deg
        float lw = 0.006;            // line half-width (rad)
        float dTheta = abs(theta - floor(theta / s + 0.5) * s);
        float par = 1.0 - smoothstep(0.0, lw, dTheta);
        float dPhi = abs(phi_angle - floor(phi_angle / s + 0.5) * s);
        float merW = lw / max(sin(theta), 0.15);
        float mer = 1.0 - smoothstep(0.0, merW, dPhi);
        float horiz = 1.0 - smoothstep(0.0, lw * 1.6, abs(theta - PI * 0.5));
        vec3 gridCol = mix(vec3(0.55, 0.85, 1.0), vec3(1.0, 0.85, 0.4), horiz);
        color = mix(color, gridCol, max(max(par, mer), horiz) * 0.5);
    }

    // Subtle lighting for 3D depth perception
    // Light from camera direction for rim-like effect
    float NdotV = abs(dot(normal, -rd));
    float lighting = mix(0.85, 1.0, NdotV);
    color *= lighting;

    outColor = vec4(color, opacity);
}
