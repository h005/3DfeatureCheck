#version 120
// Input vertex data, different for all executions of this shader.
attribute vec3 vertexPosition_modelspace;

// Output data ; will be interpolated for each fragment.
varying vec2 UV;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;

void main()
{
	vec4 viewSpacePos = mvMatrix * vec4(vertexPosition_modelspace, 1);
	gl_Position = projMatrix * viewSpacePos;
        //gl_Color = vec4(0.0,1.0,0.0,1.0);
        gl_FrontColor = vec4(0.0,0.0,1.0,1.0);
}
