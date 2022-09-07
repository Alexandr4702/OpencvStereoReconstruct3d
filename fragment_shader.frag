#ifdef GL_ES
// Set default precision to medium
precision mediump int;
precision mediump float;
#endif
varying vec4 color;


//! [0]
void main()
{
    // Set fragment color from texture
    gl_FragColor = color;
}
//! [0]

