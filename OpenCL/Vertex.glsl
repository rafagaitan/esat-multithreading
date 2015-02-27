//#version 400 compatibility

//layout(location=0) in vec4 vertexPosition;

attribute vec4 vertexPosition;

void main()
{
	gl_Position =  gl_ModelViewProjectionMatrix * vertexPosition;
}
