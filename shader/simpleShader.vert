#version 120
// Input vertex data, different for all executions of this shader.
attribute vec2 vertexUV;
attribute vec3 vertexPosition_modelspace;
attribute vec3 vertexNormal_modelspace;

// Output data ; will be interpolated for each fragment.
varying vec2 UV;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 normalMatrix;

void main()
{
	vec4 viewSpacePos = mvMatrix * vec4(vertexPosition_modelspace, 1);
	gl_Position = projMatrix * viewSpacePos;
	UV = vertexUV;
}
