varying highp vec2 v_textcoord;
uniform sampler2D u_texture;

uniform int r_id;

uniform highp vec3 u_color;
uniform highp vec3 u_lightPosition;
uniform highp float u_lightPower; // сила освещения
varying highp vec4 v_position;    // Позиция в View Space (из вершинника)
varying highp vec3 v_normal;      // Интерполированная нормаль (для вокселей)
varying highp vec3 v_color;       // Цвет из вершинника (включая радугу)
varying float v_unknown;

void main(void)
{
    // --- КАРТА ВЫСОТ ---
    if (r_id == 4) {
        if (v_unknown > 0.5) discard;

        // 1. Аппаратный расчет нормали к рельефу в пространстве вида (View Space)
        // На основе изменения экранных координат соседних фрагментов
        vec3 dx = dFdx(v_position.xyz);
        vec3 dy = dFdy(v_position.xyz);
        vec3 actualNormal = normalize(cross(dx, dy));

        // Гарантируем, что нормаль всегда развернута к экрану (к камере)
        if (actualNormal.z < 0.0) actualNormal = -actualNormal;

        // 2. Вектор освещения
        vec3 lightVec = vec3(0.0, 0.0, 1.0);

        // 3. Диффузное освещение по закону Ламберта
        float diff = max(0.0, dot(actualNormal, lightVec));

        // 4. Эмбиент (фоновое освещение), чтобы теневые склоны не были абсолютно черными
        float ambient = 0.25;

        // 5. Итоговый цвет (смешиваем радужную сетку, силу света, диффуз и эмбиент)
        gl_FragColor = vec4(v_color, 1.0) * u_lightPower * (ambient + diff);
    }

    // --- ЛИНИИ ---
    if(r_id == 3) {
        gl_FragColor = vec4(v_color, 1.0); // Альфа-канал лучше сделать 1.0, чтобы линии не были прозрачными
    }

    // --- МАРКЕРЫ / КУБЫ ---
    if(r_id == 2) {
        gl_FragColor = vec4(u_color, 1.0);
    }

    // --- ВОКСЕЛИ ---
    if(r_id == 1) {
        // Вектор освещения (из фрагмента к источнику света)
        vec3 lightVec = normalize(v_position.xyz - u_lightPosition);

        // Для вокселей оставляем честную интерполированную нормаль v_normal
        float diff = max(0.0, dot(normalize(v_normal), -lightVec));
        float ambient = 0.2;

        gl_FragColor = vec4(v_color, 1.0) * u_lightPower * (ambient + diff);
    }

    // --- КАРТА СТОИМОСТИ (COST MAP) / ТЕКСТУРЫ ---
    if(r_id == 0) {
        gl_FragColor = texture2D(u_texture, v_textcoord);
    }
}
