#version 450

layout(location = 0) in float inSpeed;
layout(location = 1) in float inType;
layout(location = 2) in float inVisualRadius;
layout(location = 3) in vec3 inSphereWorldPos;
layout(location = 4) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 1, binding = 0) uniform sampler2DArray planetTex;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    float pointScale;
    float hasPlanetTextures;
    float tanHalfFov;
    float aspectRatio;
    vec4 camRight;    // xyz = right, w = camPos.x
    vec4 camUp;       // xyz = up,    w = camPos.y
    vec4 camForward;  // xyz = fwd,   w = camPos.z
    float simTime;    // simulation time for cosmetic rotation
    float showDarkMatter;
};

// Visual rotation rates (real proportions scaled down ~50× for aesthetics)
// Negative = retrograde rotation
float rotationRate(int bodyType) {
    if (bodyType == 0) return   0.3;   // Sun
    if (bodyType == 1) return   0.12;  // Mercury (slow spinner)
    if (bodyType == 2) return  -0.03;  // Venus (very slow, retrograde)
    if (bodyType == 3) return   7.3;   // Earth
    if (bodyType == 4) return   7.1;   // Mars (~similar to Earth)
    if (bodyType == 5) return  17.6;   // Jupiter (fast spinner)
    if (bodyType == 6) return  16.5;   // Saturn
    if (bodyType == 7) return -10.2;   // Uranus (retrograde)
    if (bodyType == 8) return  10.9;   // Neptune
    // Moons: gentle rotation
    return 1.0;
}

vec3 speedToColor(float speed) {
    float t = clamp(speed / 20.0, 0.0, 1.0);
    vec3 blue   = vec3(0.2, 0.4, 1.0);
    vec3 yellow = vec3(1.0, 0.9, 0.3);
    vec3 red    = vec3(1.0, 0.2, 0.1);
    if (t < 0.5) return mix(blue, yellow, t * 2.0);
    else         return mix(yellow, red, (t - 0.5) * 2.0);
}

vec3 solarBodyColor(float bodyType) {
    int t = int(bodyType + 0.5);
    if (t == 0) return vec3(1.0, 0.85, 0.2);
    if (t == 1) return vec3(0.7, 0.65, 0.6);
    if (t == 2) return vec3(0.9, 0.85, 0.6);
    if (t == 3) return vec3(0.2, 0.5, 1.0);
    if (t == 4) return vec3(0.9, 0.4, 0.2);
    if (t == 5) return vec3(0.8, 0.7, 0.5);
    if (t == 6) return vec3(0.9, 0.8, 0.5);
    if (t == 7) return vec3(0.5, 0.8, 0.9);
    if (t == 8) return vec3(0.3, 0.4, 0.9);
    if (t == 9) return vec3(0.5, 0.5, 0.45);
    if (t == 10) return vec3(0.4, 0.45, 0.55);
    if (t == 11) return vec3(0.7, 0.7, 0.7);
    if (t == 12) return vec3(0.9, 0.85, 0.3);
    if (t == 13) return vec3(0.85, 0.9, 0.95);
    if (t == 14) return vec3(0.6, 0.55, 0.5);
    if (t == 15) return vec3(0.45, 0.4, 0.35);
    if (t == 16) return vec3(0.9, 0.7, 0.3);
    if (t == 17) return vec3(0.95, 0.95, 0.95);
    if (t == 18) return vec3(0.6, 0.7, 0.8);
    return vec3(0.6, 0.6, 0.6);
}

void main() {
    vec2 coord = inUV;
    float r2 = dot(coord, coord);

    vec3 color;
    float alpha;

    if (inVisualRadius > 0.0) {
        // ── Spherical body: perspective-correct ray-sphere intersection ──

        // Camera position (packed in .w of basis vectors)
        vec3 camP = vec3(camRight.w, camUp.w, camForward.w);
        vec3 spherePos = inSphereWorldPos;
        float R = inVisualRadius;

        // Ray direction from actual screen position (independent of billboard UV).
        // This allows the billboard to be shifted/resized without affecting the ray.
        float vpH = pointScale * 2.0;
        float vpW = vpH * aspectRatio;
        vec2 fragNDC = gl_FragCoord.xy / vec2(vpW, vpH) * 2.0 - 1.0;
        vec3 rayDir = normalize(
            camForward.xyz
            + camRight.xyz * fragNDC.x * tanHalfFov * aspectRatio
            + camUp.xyz    * fragNDC.y * tanHalfFov
        );

        // Ray-sphere intersection: |O + t*D - C|² = R²
        vec3 oc = camP - spherePos;
        float b = dot(rayDir, oc);
        float c = dot(oc, oc) - R * R;
        float disc = b * b - c;

        int bodyType = int(inType + 0.5);
        bool hitSphere = (disc >= 0.0);
        float tSphere = 1e30;
        if (hitSphere) {
            float sqrtDisc = sqrt(disc);
            tSphere = -b - sqrtDisc;
            if (tSphere < 0.0) tSphere = -b + sqrtDisc;
            if (tSphere < 0.0) hitSphere = false;
        }

        // ── Saturn ring: ray-plane intersection with equatorial plane ──
        bool hitRing = false;
        float tRing = 1e30;
        vec3 ringHitPoint;
        float ringU = 0.0;  // texture coordinate (0=inner, 1=outer)
        if (bodyType == 6 && hasPlanetTextures > 0.5) {
            // Ring plane: Y = spherePos.y (ecliptic plane through Saturn)
            vec3 planeN = vec3(0.0, 1.0, 0.0);
            float denom = dot(rayDir, planeN);
            if (abs(denom) > 1e-6) {
                float tPlane = dot(spherePos - camP, planeN) / denom;
                if (tPlane > 0.0) {
                    vec3 hp = camP + rayDir * tPlane;
                    float dist2D = length(hp.xz - spherePos.xz);
                    // Ring spans 1.27 to 2.40 Saturn radii
                    float rInner = R * 1.27;
                    float rOuter = R * 2.40;
                    if (dist2D >= rInner && dist2D <= rOuter) {
                        hitRing = true;
                        tRing = tPlane;
                        ringHitPoint = hp;
                        ringU = (dist2D - rInner) / (rOuter - rInner);
                    }
                }
            }
        }

        // Choose closest hit: sphere or ring
        if (!hitSphere && !hitRing) discard;

        vec3 hitPoint;
        if (hitSphere && (!hitRing || tSphere <= tRing)) {
            // ── Sphere hit ──
            hitPoint = camP + rayDir * tSphere;
            vec3 worldN = (hitPoint - spherePos) / R;
            vec3 viewDir = normalize(camP - hitPoint);
            float NdotV = max(dot(worldN, viewDir), 0.0);

            if (hasPlanetTextures > 0.5) {
                float u = atan(worldN.x, worldN.z) / (2.0 * 3.14159265) + 0.5;
                // Cosmetic rotation: offset longitude by time × rotation rate
                u += simTime * rotationRate(bodyType);
                u = fract(u);  // wrap to [0, 1]
                float v = asin(clamp(worldN.y, -1.0, 1.0)) / 3.14159265 + 0.5;
                vec3 texColor = texture(planetTex, vec3(u, v, inType)).rgb;
                float limbFade = smoothstep(0.0, 0.15, NdotV);
                color = mix(solarBodyColor(inType), texColor, limbFade);
            } else {
                color = solarBodyColor(inType);
            }

            if (bodyType == 0) {
                float edgeFade = sqrt(1.0 - NdotV * NdotV);
                float glow = 1.0 - edgeFade * 0.3;
                color *= glow;
                alpha = 1.0;
            } else {
                vec3 sunDir = normalize(-spherePos);
                float diffuse = max(dot(worldN, sunDir), 0.0);
                vec3 halfVec = normalize(sunDir + viewDir);
                float spec = pow(max(dot(worldN, halfVec), 0.0), 32.0);
                float ambient = (hasPlanetTextures > 0.5) ? 0.01 : 0.08;
                float lighting = ambient + 0.92 * diffuse + 0.3 * spec;
                color *= lighting;
                alpha = 1.0;
            }
        } else {
            // ── Ring hit ──
            hitPoint = ringHitPoint;
            // Sample ring texture (layer 19): U = radial position, V = 0.5
            vec4 ringTex = texture(planetTex, vec3(ringU, 0.5, 19.0));
            color = ringTex.rgb;
            alpha = ringTex.a;

            // Sun lighting on ring surface
            vec3 sunDir = normalize(-spherePos);
            vec3 ringNormal = vec3(0.0, 1.0, 0.0);
            // Rings are flat — lit from both sides
            float diffuse = abs(dot(ringNormal, sunDir));
            float ambient = 0.05;
            color *= ambient + 0.95 * diffuse;

            // Shadow: check if ring point is behind Saturn (simple cylinder shadow)
            vec3 toSun = normalize(-hitPoint);
            vec3 shadowOC = hitPoint - spherePos;
            float sb = dot(toSun, shadowOC);
            float sc = dot(shadowOC, shadowOC) - R * R;
            float sDisc = sb * sb - sc;
            if (sDisc > 0.0 && (-sb - sqrt(sDisc)) > 0.0) {
                // In Saturn's shadow
                color *= 0.1;
            }

            if (alpha < 0.01) discard;
        }

        // Depth from actual hit point (correct occlusion)
        vec4 hitClip = viewProj * vec4(hitPoint, 1.0);
        gl_FragDepth = hitClip.z / hitClip.w;

    } else if (inType > 0.0) {
        // Sink / black hole
        if (r2 > 1.0) discard;
        float falloff = 1.0 - sqrt(r2);
        alpha = falloff * falloff * 2.0;
        color = vec3(1.0, 1.0, 1.0);
        gl_FragDepth = 1.0;
    } else {
        // Default: speed-based color
        if (r2 > 1.0) discard;
        float falloff = 1.0 - sqrt(r2);
        alpha = falloff * falloff;
        color = speedToColor(inSpeed);
        gl_FragDepth = 1.0;
    }

    outColor = vec4(color * alpha, alpha);
}
