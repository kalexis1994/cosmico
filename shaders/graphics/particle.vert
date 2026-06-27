#version 450

struct Particle {
    vec4 position;    // xyz = position, w = mass
    vec4 velocity;    // xyz = velocity, w = visualRadius
    vec4 attributes;  // x = density, y = pressure, z = smoothing_length, w = type
};

layout(std430, set = 0, binding = 0) readonly buffer ParticleBuffer {
    Particle particles[];
};

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
    float lumStrength;
    float coordScale;  // 1 = comoving frame; a/aInit = physical (box inflates)
};

layout(location = 0) out float outSpeed;
layout(location = 1) out float outType;
layout(location = 2) out float outVisualRadius;
layout(location = 3) out vec3 outSphereWorldPos;
layout(location = 4) out vec2 outUV;
layout(location = 5) out float outLum;   // local overdensity δ (PM luminosity)
layout(location = 6) out float outMass;  // particle mass (accreted = bright)

// 6 vertices for a quad (2 triangles)
const vec2 QUAD_CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
);

void main() {
    int particleIdx = gl_InstanceIndex;
    int cornerIdx = gl_VertexIndex;

    Particle p = particles[particleIdx];
    vec2 corner = QUAD_CORNERS[cornerIdx];

    float particleType = p.attributes.w;

    // Skip dark matter particles (type < 0) unless toggle is on
    if (particleType < 0.0 && showDarkMatter < 0.5) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        outSpeed = 0.0;
        outType = particleType;
        outVisualRadius = 0.0;
        outSphereWorldPos = vec3(0.0);
        outUV = vec2(0.0);
        outLum = 0.0;
        outMass = 0.0;
        return;
    }

    vec3 camP = vec3(camRight.w, camUp.w, camForward.w);
    // Comoving→physical: positions are centred on the origin, so scaling them
    // by a(t)/aInit inflates the box and makes structure recede (coordScale=1
    // leaves the comoving frame untouched).
    vec3 worldPos = p.position.xyz * coordScale;
    float smoothingRadius = p.velocity.w;

    // Compute billboard half-extent in world space
    float halfSize;

    if (smoothingRadius > 0.0) {
        // Spherical body: billboard sized to cover visible disk
        float dist = length(worldPos - camP);
        float R = smoothingRadius;
        // Saturn (type 6): extend billboard to cover rings (2.4× planet radius)
        float margin = (int(particleType + 0.5) == 6) ? 2.6 : 1.3;
        if (dist > R * 1.001) {
            float ratio = R / dist;
            halfSize = R / sqrt(1.0 - ratio * ratio);
            halfSize *= margin;
        } else {
            // Camera inside or touching sphere: cover full hemisphere
            halfSize = R * 50.0;
        }
        outVisualRadius = smoothingRadius;
    } else {
        // Regular particle: fixed world size with perspective shrinking + distance fade
        float dist = max(length(worldPos - camP), 0.001);
        float massScale;
        int iType = int(particleType + 0.5);
        if (iType == 9 || iType == 10) {
            // Asteroid/KBO
            massScale = 0.002;
        } else if (particleType > 0.0) {
            massScale = clamp(p.position.w * 4.0, 0.3, 4.0);
        } else {
            massScale = clamp(p.position.w * 2.0, 0.002, 1.0);
        }
        // World size of 1 pixel at this distance
        float pxWorld = dist * tanHalfFov / (2.0 * pointScale);
        // Natural pixel count for this particle
        float naturalPx = massScale / pxWorld;
        // Render at least 1px for rasterization, max 6px
        halfSize = clamp(massScale, pxWorld, pxWorld * 6.0);
        // Sub-pixel alpha: fade out when natural size < 1 pixel
        outVisualRadius = clamp(naturalPx, 0.0, 1.0);
    }

    // Billboard quad in world space.
    // When the sphere is close, the billboard center may be near/behind the
    // camera plane causing clipping.  Since the fragment shader computes rays
    // from gl_FragCoord (screen-space), the billboard only needs to COVER the
    // right pixels — its position doesn't affect the ray direction.
    // So for close spheres we emit a full-screen clip-space quad.
    bool fullScreen = false;
    if (smoothingRadius > 0.0) {
        float frontDist = dot(worldPos - camP, camForward.xyz);
        // Only go fullscreen when sphere center is very near or behind camera
        if (frontDist < smoothingRadius * 1.5) {
            fullScreen = true;
        }
    }

    if (fullScreen) {
        gl_Position = vec4(corner.x, corner.y, 0.5, 1.0);
    } else {
        vec3 right = camRight.xyz;
        vec3 up = camUp.xyz;
        vec3 vertexPos = worldPos + halfSize * (corner.x * right + corner.y * up);
        gl_Position = viewProj * vec4(vertexPos, 1.0);
    }

    // UV in [-1,1]: maps billboard corners to ray offsets
    // corner.y=+1 (world up via camUp) renders at bottom of framebuffer
    // (no proj Y-flip), matching gl_PointCoord coord.y=+1 at bottom
    outUV = vec2(corner.x, corner.y);

    outSpeed = length(p.velocity.xyz);
    outType = particleType;
    outLum = p.attributes.x;   // local overdensity stored by the PM step
    outMass = p.position.w;
    // outVisualRadius already set: smoothingRadius (sphere) or sub-pixel alpha (particle)
    outSphereWorldPos = worldPos;
}
