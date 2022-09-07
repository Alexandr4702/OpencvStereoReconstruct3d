#ifdef GL_ES
precision mediump int;
precision mediump float;
#endif

uniform mat4 mvp_matrix;

attribute vec4 a_position;

varying vec4 color;
void main()
{
    gl_Position = mvp_matrix * a_position;
    color=vec4(clamp(a_position[0],0.1,1.0),clamp(a_position[1],0.1,1.0),clamp(a_position[2],0.1,1.0),1);
}
