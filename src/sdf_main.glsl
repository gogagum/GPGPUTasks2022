// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// see https://iquilezles.org/articles/distfunctions/
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// smooth minimum function to create gradual transitions between SDFs
// https://iquilezles.org/articles/smin/
float smoothmin(float d0, float d1, float k)
{
    return -log(1. / exp(k*d0) + 1. / exp(k*d1))/k;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

vec4 sdBody(vec3 p, vec3 color)
{
    float d = 1e10;

    // body, two spheres with smoothmin
    float d_low = sdSphere((p - vec3(0.0, 0.45, -0.7)), 0.35);
    float d_high = sdSphere((p - vec3(0.0, 0.75, -0.7)), 0.25);

    d = smoothmin(d_low, d_high, 12.);


    // hands, two capsules, can wave with lazycos
    float leftHand = sdCapsule(p,
                               vec3(0.3, 0.65, -0.7),
                               vec3(0.4, 0.55, -0.7),
                               0.06);

    vec3 rightHandEnd = vec3(-0.4, 0.65 + 0.15 * lazycos(8.0 * iTime), -0.7);

    float rightHand = sdCapsule(p,
                                vec3(-0.3, 0.65, -0.7),
                                rightHandEnd,
                                0.06);

    float hands = min(leftHand, rightHand);

    d = min(hands, d);

    // legs, two capsules
    float leftLeg = sdCapsule(p,
                              vec3(-0.15, 0.55, -0.7),
                              vec3(-0.15, 0.05, -0.7),
                              0.07);

    float rightLeg = sdCapsule(p,
                              vec3(0.15, 0.55, -0.7),
                              vec3(0.15, 0.05, -0.7),
                              0.07);

    float legs = min(leftLeg, rightLeg);

    d = min(legs, d);

    // return distance and color
    return vec4(d, color);
}

vec4 sdEyeBall(vec3 p)
{
    float d0 = sdSphere(p - vec3(0.0, 0.7, -0.5), 0.2);

    // return distance and color
    return vec4(d0, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p)
{
    float d0 = sdSphere(p - vec3(0.0, 0.7, -0.30), 0.06);

    // return distance and color
    return vec4(d0, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p)
{
    float d0 = sdSphere(p - vec3(0.0, 0.7, -0.39), 0.12);

    // return distance and color
    return vec4(d0, vec3(0.0, 1.0, 1.0));

}

vec4 sdEye(vec3 p)
{
    vec4 res = sdEyeBall(p);

    vec4 ep = sdEyePupil(p);

    if (res.x > ep.x) {
        res = ep;
    }

    vec4 ei = sdEyeIris(p);
    if (res.x > ei.x) {
        res = ei;
    }


    return res;
}

vec4 sdMonster(vec3 p, vec3 color)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    vec4 res = sdBody(p, color);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p - vec3(0.5, 0., 0.), vec3(0., 1., 0.));
    vec4 m2 = sdMonster(p + vec3(0.5, 0., 0.), vec3(0., 0., 1.));

    if (m2.x < res.x) {
        res = m2;
    }

    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{

    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);


    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}
