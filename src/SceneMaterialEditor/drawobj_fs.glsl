#version 450
precision highp float;
precision highp int;

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 uv;

uniform sampler2D s0;	// albedo.
uniform vec4 color;

uniform bool hasAlbedo = false;

uniform int materialId;
uniform bool isSelected = false;
uniform float time;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outAttrib;

void main()
{
	if (hasAlbedo) {
		outColor = texture2D(s0, uv) * color;
	}
	else {
		// TODO
		outColor = color;
	}

	if (isSelected) {
		vec3 complementaryColor = max(vec3(1) - outColor.rgb, vec3(0));
		outColor.rgb = mix(outColor.rgb, complementaryColor, time);
	}

	// NOTE
	// 0 は何もない扱いにする...
	outAttrib = vec4((materialId + 1) / 255.0f, 0, 0, 1);
}
