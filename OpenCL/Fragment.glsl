uniform vec4 diffuseColor;
uniform vec4 lightColor;
uniform vec4 ambientColor;

void main()
{
	gl_FragColor = ambientColor + diffuseColor * lightColor;
}