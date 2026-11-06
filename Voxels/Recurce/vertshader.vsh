attribute highp vec4 a_position;
attribute highp vec2 a_textcoord;
attribute highp vec3 a_normal;

uniform highp mat4 u_projectionMatrix;
uniform highp mat4 u_viewMatrix;
uniform highp mat4 u_modelMatrix;

attribute vec3 position_vector;
attribute vec3 color_position;

attribute float a_position_vector;
attribute float a_color_position;

uniform int p_id;
uniform int u_apply_map_offset;
uniform vec3 u_map_offset;

varying highp vec4 v_position;     // Позиция в пространстве вида
varying highp vec2 v_textcoord;
varying highp vec3 v_color;
varying highp vec3 v_normal;      // Передается для вокселей
varying float v_unknown;

uniform int   u_wire;     // 0 - заливка, 1 - каркас
uniform float u_hMin;
uniform float u_hMax;

uniform sampler2D u_heightTexture;
attribute float a_parity;

// Распаковка цвета для вокселей
vec4 unpack_color(float packed_float) {
    uint packed_uint = floatBitsToUint(packed_float);
    float r = float((packed_uint >> 24) & 0xFF) / 255.0;
    float g = float((packed_uint >> 16) & 0xFF) / 255.0;
    float b = float((packed_uint >> 8) & 0xFF) / 255.0;
    float a = float(packed_uint & 0xFF) / 255.0;
    return vec4(r, g, b, a);
}

// Распаковка позиции для вокселей
vec3 unpack_position(float packed_float) {
    uint packed_uint = floatBitsToUint(packed_float);
    vec3 pos_vec;
    pos_vec.z = -float(packed_uint / (400 * 100));
    uint remaining = packed_uint % (400 * 100);
    pos_vec.y = -float(remaining / 400);
    pos_vec.x = float(packed_uint % 400);
    return pos_vec;
}

// Радужный градиент высот для карты
vec3 rainbow(float t) {
     t = t * 0.5 + 0.5;
    float h = t * 5.0 + 1.0;
    int i = int(floor(h));
    float f = h - float(i);
    if ((i & 1) == 0) f = 1.0 - f;
    float n = 1.0 - f;
    if (i <= 1) return vec3(n, 0.0, 1.0);
    if (i == 2) return vec3(0.0, n, 1.0);
    if (i == 3) return vec3(0.0, 1.0, n);
    if (i == 4) return vec3(n, 1.0, 0.0);
    return vec3(1.0, n, 0.0);
}

vec3 effective_map_offset() {
    return (u_apply_map_offset == 1) ? u_map_offset : vec3(0.0, 0.0, 0.0);
}

void main(void)
{
    vec3 map_off = effective_map_offset();

    // --- КАРТА ВЫСОТ ---
    if (p_id == 4) {
        // Чтение упакованного float (два int16_t в одном)
        float packed = texture2D(u_heightTexture, a_textcoord).r;
        uint p = floatBitsToUint(packed);

        // Выбор нужных 16 бит (int16_t, знаковый)
        uint raw = (a_parity < 0.5) ? ((p >> 16) & 0xFFFFu) : (p & 0xFFFFu);
        int h_mm = int(raw);
        if (h_mm >= 32768) h_mm -= 65536;   // знаковое приведение

        v_unknown = (h_mm == -32768) ? 1.0 : 0.0;

        float h = (h_mm == -32768) ? 0.0 : float(h_mm) * 0.005f;


        // Трансформация
        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.y += h;
        worldPos.xyz += map_off;
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
        v_position = u_viewMatrix * worldPos;

        // Цвет
        float h_color = h / 5.0f;
        float t01 = (u_hMax > u_hMin) ? ((h_color - u_hMin) / (u_hMax - u_hMin)) : 0.0;
        float t = t01 * 2.0 - 1.0;  // [0, 1] -> [-1, 1]
        v_color = (u_wire == 1) ? vec3(0.0, 0.0, 0.0) : rainbow(t);
        v_normal = vec3(0.0, 1.0, 0.0);
    }

    // --- ЛИНИИ ---
    if(p_id == 3) {
        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += map_off;
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
        v_color = color_position;
    }

    // --- МАРКЕРЫ / КУБЫ ---
    if(p_id == 2) {
        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += map_off;
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
    }

    // --- ВОКСЕЛИ ---
    if(p_id == 1) {
        vec3 local_color = unpack_color(a_color_position).xyz;
        vec3 pos_vec = unpack_position(a_position_vector);

        if(local_color.x == 0.0f && local_color.y == 0.0f && local_color.z == 0.0f) {
            local_color = vec3(1.0, 1.0, 1.0);
            float v = -pos_vec.y;
            float vmin = 29.0f;
            float vmax = 79.0f;
            v = clamp(v, vmin, vmax);
            float dv = vmax - vmin;

            if (v < (vmin + 0.25f * dv)) {
                local_color.x = 0.0f;
                local_color.y = 4.0f * (v - vmin) / dv;
            }
            else if (v < (vmin + 0.5f * dv)) {
                local_color.x = 0.0;
                local_color.z = 1.0f + 4.0f * (vmin + 0.25f * dv - v) / dv;
            }
            else if (v < (vmin + 0.75f * dv)) {
                local_color.x = 4.0f * (v - vmin - 0.5f * dv) / dv;
                local_color.z = 0.0f;
            }
            else {
                local_color.y = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv;
                local_color.z = 0.0f;
            }
        }

        mat4 position_matrix = mat4(1.0, 0.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0,
                                    pos_vec.x, pos_vec.y, pos_vec.z, 1.0);

        gl_Position = u_projectionMatrix * u_viewMatrix * position_matrix * a_position;
        mat4 mv_matrix = u_viewMatrix * position_matrix;

        // Для вокселей сохраняем классическую интерполяцию переданных нормалей
        v_normal = normalize(vec3(mv_matrix * vec4(a_normal, 0.0)));
        v_position = mv_matrix * a_position;
        v_color = local_color;
    }

    // --- ТЕКСТУРИРОВАННЫЙ КВАД (COST MAP) ---
    if (p_id == 0) {
        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += map_off;
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
        v_textcoord = a_textcoord;
    }
}
