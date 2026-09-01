attribute highp vec4 a_position;

attribute highp vec2 a_textcoord;

attribute highp vec3 a_normal;

uniform highp mat4 u_projectionMatrix;

uniform highp mat4 u_viewMatrix;

uniform highp mat4 u_modelMatrix;

vec3 position_vector;

vec3 color_position;

attribute uint voxels;

uniform int p_id;

varying highp vec4 v_position;

varying highp vec2 v_textcoord;

varying highp vec3 v_color;

varying highp vec3 v_normal;


void main(void)

{
    int instanceID = gl_InstanceID;

    color_position.x = 1.0f; color_position.y = 1.0f; color_position.z = 1.0f;// white

    float dv; float v = -position_vector.y; float vmin = 29.0f; float vmax = 79.0f;

    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25f * dv)) { color_position.x = 0.0f; color_position.y = 4.0f * (v - vmin) / dv; }
    else if (v < (vmin + 0.5f * dv)) { color_position.x = 0.0; color_position.z = 1.0f + 4.0f * (vmin + 0.25 * dv - v) / dv; }
    else if (v < (vmin + 0.75f * dv)) { color_position.x = 4.0f * (v - vmin - 0.5f * dv) / dv; color_position.z = 0.0f; }
    else {color_position.y = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv; color_position.z = 0.0f;}

    if(voxels == 1) {

    position_vector.z = instanceID  / (400 * 100);
    uint remaining = instanceID  % (400 * 100);
    position_vector.y = remaining / 400;
    position_vector.x = instanceID  % 400;

    }


    mat4 position_matrix = mat4 (1.0,            0.0,                  0.0,               0.0,
                                 0.0,            1.0,                  0.0,               0.0,
                                 0.0,            0.0,                  1.0,               0.0,
                                 position_vector.x, position_vector.y, position_vector.z, 1.0);

     gl_Position = u_projectionMatrix * u_viewMatrix * position_matrix * a_position;

     mat4 mv_matrix = u_viewMatrix * position_matrix;

     v_normal = normalize(vec3(mv_matrix * vec4(a_normal, 0.0)));

     v_position = mv_matrix * a_position;

     v_color = color_position;
    }

    if (p_id == 0) { gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * a_position;

     v_textcoord = a_textcoord; }

}
