#version 120

//in vec3 vColor

//out vec4 gl_FragColor

void main(){
	// Output color = color of the texture at the specified UV
//        gl_FragColor = vec4 (vColor, 1.0);
    gl_FragColor = vec4 (0.0, 1.0, 0.0, 1.0);
}
