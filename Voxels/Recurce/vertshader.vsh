// vertshader.vsh
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
uniform vec3 u_map_offset;  // Новый uniform для смещения карты

varying highp vec4 v_position;
varying highp vec2 v_textcoord;
varying highp vec3 v_color;
varying highp vec3 v_normal;

vec4 unpack_color(float packed_float) {
    uint packed_uint = floatBitsToUint(packed_float);
    float r = float((packed_uint >> 24) & 0xFF) / 255.0;
    float g = float((packed_uint >> 16) & 0xFF) / 255.0;
    float b = float((packed_uint >> 8) & 0xFF) / 255.0;
    float a = float(packed_uint & 0xFF) / 255.0;
    return vec4(r, g, b, a);
}

vec3 unpack_position(float packed_float) {
    uint packed_uint = floatBitsToUint(packed_float);

    // Распаковка координат (x, y, z) из одного числа
    vec3 position_vector;
    position_vector.z = -float(packed_uint / (400 * 100));
    uint remaining = packed_uint % (400 * 100);
    position_vector.y = -float(remaining / 400);
    position_vector.x = float(packed_uint % 400);

    return position_vector;
}

void main(void)
{
    if(p_id == 3) {

        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += u_map_offset;  // Применяем смещение
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
        v_color = color_position;
    }

    if(p_id == 2) {

        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += u_map_offset;  // Применяем смещение
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
    }

    if(p_id == 1) {
        // Для вокселей
        color_position = unpack_color(a_color_position).xyz;
        position_vector = unpack_position(a_position_vector);

        if(color_position.x == 0.0f && color_position.y == 0.0f && color_position.z == 0.0f) {
            color_position.x = 1.0f; color_position.y = 1.0f; color_position.z = 1.0f;
            float dv; float v = -position_vector.y; float vmin = 29.0f; float vmax = 79.0f;
            if (v < vmin) v = vmin;
            if (v > vmax) v = vmax;
            dv = vmax - vmin;
            if (v < (vmin + 0.25f * dv)) {
                color_position.x = 0.0f;
                color_position.y = 4.0f * (v - vmin) / dv;
            }
            else if (v < (vmin + 0.5f * dv)) {
                color_position.x = 0.0;
                color_position.z = 1.0f + 4.0f * (vmin + 0.25f * dv - v) / dv;
            }
            else if (v < (vmin + 0.75f * dv)) {
                color_position.x = 4.0f * (v - vmin - 0.5f * dv) / dv;
                color_position.z = 0.0f;
            }
            else {
                color_position.y = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv;
                color_position.z = 0.0f;
            }
        }

        mat4 position_matrix = mat4 (1.0, 0.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0,
                                     position_vector.x, position_vector.y, position_vector.z, 1.0);

        gl_Position = u_projectionMatrix * u_viewMatrix * position_matrix * a_position;
        mat4 mv_matrix = u_viewMatrix * position_matrix;
        v_normal = normalize(vec3(mv_matrix * vec4(a_normal, 0.0)));
        v_position = mv_matrix * a_position;
        v_color = color_position;
    }

    if (p_id == 0) {

        vec4 worldPos = u_modelMatrix * a_position;
        worldPos.xyz += u_map_offset;  // Применяем смещение
        gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
        v_textcoord = a_textcoord;
    }
}
