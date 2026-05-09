#version 450

layout(set = 0, binding = 0) uniform sampler3D volumeTex;

layout(push_constant) uniform PC {
    mat4 invViewProj;
    vec4 cameraPos;
    vec4 volumeMin;
    vec4 volumeMax;
    float opacityScale;
    float stepSize;
    int numSteps;
};

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// Ray-box intersection (slab method)
bool intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out float tNear, out float tFar) {
    vec3 invRd = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invRd;
    vec3 t1 = (bmax - ro) * invRd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    tNear = max(max(tmin.x, tmin.y), tmin.z);
    tFar  = min(min(tmax.x, tmax.y), tmax.z);
    return tNear <= tFar && tFar > 0.0;
}

// Hot colormap: black -> red -> orange -> yellow -> white
vec3 densityToColor(float d) {
    vec3 c;
    c.r = smoothstep(0.0, 0.35, d);
    c.g = smoothstep(0.25, 0.65, d);
    c.b = smoothstep(0.5, 0.95, d);
    return c;
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

    // Intersect ray with volume bounding box
    float tNear, tFar;
    if (!intersectBox(ro, rd, volumeMin.xyz, volumeMax.xyz, tNear, tFar)) {
        outColor = vec4(0.0);
        return;
    }

    tNear = max(tNear, 0.0);

    vec3 boxSize = volumeMax.xyz - volumeMin.xyz;

    // Front-to-back compositing
    vec4 accumulated = vec4(0.0);

    for (int i = 0; i < numSteps; i++) {
        if (accumulated.a >= 0.99) break;

        float t = tNear + (float(i) + 0.5) * stepSize;
        if (t > tFar) break;

        vec3 pos = ro + rd * t;

        // Map world position to [0,1] UVW texture coordinates
        vec3 uvw = (pos - volumeMin.xyz) / boxSize;

        // Sample density (0 = mean deviation, 1 = +3sigma peak)
        float raw = texture(volumeTex, uvw).r;

        // Threshold: cut values below ~0.15 to eliminate uniform fog
        // Then remap remaining range to [0,1] with power curve for contrast
        float threshold = 0.1;
        float density = max(raw - threshold, 0.0) / (1.0 - threshold);

        // Power curve to enhance peaks and suppress noise
        density = pow(density, 1.5);

        if (density < 0.001) continue;

        // Colormap
        vec3 sampleColor = densityToColor(density);

        // Opacity: proportional to density with global scale
        float sampleAlpha = density * opacityScale * stepSize;
        sampleAlpha = clamp(sampleAlpha, 0.0, 1.0);

        // Front-to-back compositing
        accumulated.rgb += (1.0 - accumulated.a) * sampleColor * sampleAlpha;
        accumulated.a   += (1.0 - accumulated.a) * sampleAlpha;
    }

    outColor = accumulated;
}
