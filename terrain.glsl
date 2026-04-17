// original: https://www.shadertoy.com/view/lsj3z3

#define M_PI 3.1415926535897932384626433832795


float terrain( in vec2 p)
{
    
    const float heightScale = 1.0;
   	const bool flip = true;
    
    p = p*0.1;
    float scale = 1.0;
    //float scale = max(0.0, mix(1.0, 0.0, max(1.0, 2.0*max(abs(p.x), abs(p.y))) - 1.0));
    p = p+vec2(0.5);
    //scale = -2.0*scale*scale*scale + 3.0*scale*scale;
    if (p.x < 0.0 || p.x >= 1.0 || p.y < 0.0 || p.y >= 1.0)
        return 0.0;
   	p = mix(vec2(0.5, 0.0), vec2(1.0, 0.5), p);
    float val = texture( iChannel0, p, 0.0 ).x;
    if (flip)
        val = 1.0 - val;
   	//val = (val-0.5)*5.0;
	return val * scale * heightScale;
}

float map( in vec3 p )
{
    return p.y - terrain(p.xz);
}

vec3 calcNormal( in vec3 pos, float t )
{
	float e = 0.001;
	e = 0.0001*t;
    vec3  eps = vec3(e,0.0,0.0);
    vec3 nor;
    nor.x = map(pos+eps.xyy) - map(pos-eps.xyy);
    nor.y = map(pos+eps.yxy) - map(pos-eps.yxy);
    nor.z = map(pos+eps.yyx) - map(pos-eps.yyx);
    return normalize(nor);
}

float intersect( in vec3 ro, in vec3 rd )
{
    const float maxd = 40.0;
    const float precis = 0.001;
    float t = 0.0;
    for( int i=0; i<256; i++ )
    {
        float h = map( ro+rd*t );
        if( abs(h)<precis || t>maxd ) break;
        t += h*0.5;
    }
    return (t>maxd)?-1.0:t;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	float time = iTime;
    float rotTimeScale = 0.3;
    
    //vec4 m = iMouse / iResolution.xxxx;
    
    float mousePos = -iMouse.x/iResolution.x * 2.0 * M_PI * 8.0;
    time = mousePos;
	
	// Normal set up code for ray trace.
	vec2 xy = fragCoord.xy / iResolution.xy;
	vec2 s = (-1.0 + 2.0* xy) * vec2(iResolution.x/iResolution.y, 1.0);
    
    vec3 planeNormal = normalize(vec3(0.0, 1.0, 0.0));
    vec3 planeUp = abs(planeNormal.y) < 0.5
        ? vec3(0.0, 1.0, 0.0)
        : vec3(0.0, 0.0, 1.0);
   	vec3 planeSide = normalize(cross(planeNormal, planeUp));
    planeUp = cross(planeSide, planeNormal);
	
    float dist = 8.0;
    float height = 2.0;
    
	vec3 ro = vec3(sin(time * rotTimeScale) * dist, height, cos(time * rotTimeScale) * dist);
    ro.y += (planeSide*ro.x + planeNormal*ro.y + planeUp*ro.z).y;
	vec3 cd = normalize(-ro);
	vec3 cu = vec3(0.0, 1.0, 0.0);
	vec3 cr = normalize(cross(cd, cu));
	cu = normalize(cross(cr, cd));

	vec3 rd = normalize( s.x*cr + s.y*cu + 2.0*cd );

	vec3 col = vec3(0.7, 0.7, 0.7);
    	
    vec3 sunDir = normalize(vec3(-1, -1, -1));
    vec3 ambientLight = vec3(0.3);
    vec3 sunLight = vec3(0.9);
    
    
    // transform ray
    ro = vec3(
        dot(ro, planeSide),
        dot(ro, planeNormal),
        dot(ro, planeUp)
    );
    rd = vec3(
        dot(rd, planeSide),
        dot(rd, planeNormal),
        dot(rd, planeUp)
    );
    
    
    float t = intersect(ro, rd);
    
    if(t > 0.0)
    {	
		// Get some information about our intersection
		vec3 pos = ro + t * rd;
		vec3 normal = calcNormal(pos, t);
        
        float shadow = intersect(pos, -sunDir) > 0.0 ? 0.0 : 1.0;
		
		vec2 uvp = vec2(dot(pos, planeUp), dot(pos, planeSide));
		
		vec3 texCol = texture(iChannel1, uvp, 0.0).xyz;
		
		col = texCol * clamp(-dot(normal, sunDir), 0.0f, 1.0f) * shadow * sunLight + ambientLight; 
	}
	
	fragColor = vec4(pow(col*1.2, vec3(2.2)), 1.0);
}

