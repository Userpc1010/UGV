varying highp vec2 v_textcoord;
uniform sampler2D u_texture;

uniform int r_id;

uniform highp vec3 u_color;
uniform highp vec3 u_lightPosition;
uniform highp float u_lightPower; // сила освещения
varying highp vec4 v_position;
varying highp vec3 v_normal;
varying highp vec3 v_color;


void main(void)
{
    if(r_id == 3) gl_FragColor = vec4(v_color, 0.0);

    if(r_id == 2) gl_FragColor = vec4(u_color, 0.0);

    if(r_id == 1) { vec3 lightVec = normalize(v_position.xyz - u_lightPosition); // вектор освещения

    gl_FragColor = vec4(v_color, 0.0) * u_lightPower * max(0.0, dot(v_normal, -lightVec));}

    if(r_id == 0) gl_FragColor = texture2D(u_texture, v_textcoord);
}
