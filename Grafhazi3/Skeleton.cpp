//=============================================================================================
// Mintaprogram: Z?ld h?romsz?g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev : T�th B�lint
// Neptun : WKRJAD
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct BodyState {
    float fi;
    vec3 c, p, L;
};

//---------------------------
template<class T>
struct Dnum { // Dual numbers for automatic derivation
//---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }

    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }

    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }

    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }

    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};

// Elementary functions prepared for the chain rule as well
template<class T>
Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }

template<class T>
Dnum<T> Sin(Dnum<T> g) { return Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }

template<class T>
Dnum<T> Cos(Dnum<T> g) { return Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }

template<class T>
Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }

template<class T>
Dnum<T> Sinh(Dnum<T> g) { return Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }

template<class T>
Dnum<T> Cosh(Dnum<T> g) { return Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }

template<class T>
Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }

template<class T>
Dnum<T> Log(Dnum<T> g) { return Dnum<T>(logf(g.f), g.d / g.f); }

template<class T>
Dnum<T> Pow(Dnum<T> g, float n) {
    return Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;

//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;        // intrinsic
public:
    Camera() {
        asp = (float) windowWidth / 2 / windowHeight;
        fov = 75.0f * (float) M_PI / 180.0f;
        fp = 2;
        bp = 50;
    }

    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0, 0, 0, 1);
    }

    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
                    0, 1 / tan(fov / 2), 0, 0,
                    0, 0, -(fp + bp) / (bp - fp), -1,
                    0, 0, -2 * fp * bp / (bp - fp), 0);
    }
};

//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};

//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};

//---------------------------

//---------------------------
struct RenderState {
//---------------------------
    mat4 MVP, M, Minv, V, P;
    Material *material;
    std::vector<Light> lights;
    vec3 wEye;
};

//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;

    void setUniformMaterial(const Material &material, const std::string &name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }

    void setUniformLight(const Light &light, const std::string &name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};

float maxHeight = -1000; // global variable to store the max height of the terrain
float minHeight = 1000; // global variable to store the min height of the terrain

//---------------------------
class MapShader : public Shader {
//---------------------------
    const char *vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in float  y;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
        out float height;

		void main() {
            height = y;
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

    // fragment shader in GLSL
    const char *fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

        vec3 gradient[2] = vec3[](vec3(0, 0.5, 0), vec3(0.5, 0.3, 0.1));


		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
        uniform float maxHeight;
        uniform float minHeight;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
        in  float height;

        vec3 mapColor(){

            float normalVal = (height - minHeight)/(maxHeight - minHeight);
            return mix(gradient[0],gradient[1],normalVal+0.3);
}

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.ka;
			vec3 kd = mapColor();

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    MapShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();        // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(minHeight, "minHeight");
        setUniform(maxHeight, "maxHeight");
        setUniformMaterial(*state.material, "material");

        setUniform((int) state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};

class JumperShader : public Shader {
//---------------------------
    const char *vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

// fragment shader in GLSL
    const char *fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};


		uniform Material material;
		uniform Light[8] lights;    // light sources
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir

        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.ka;
			vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
    JumperShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

    void Bind(RenderState state) {
        Use();        // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");

        setUniform((int) state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }

};

//---------------------------
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }

    virtual void Draw() = 0;

    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};


//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    struct VertexData {
        vec3 position, normal;
        vec2 texcoord;
    };

    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }

    virtual void eval(Dnum2 &U, Dnum2 &V, Dnum2 &X, Dnum2 &Y, Dnum2 &Z) = 0;

    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        vtxData.texcoord = vec2(u, v);
        Dnum2 X, Y, Z;
        Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        eval(U, V, X, Y, Z);
        vtxData.position = vec3(X.f, Y.f, Z.f);
        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = cross(-drdU, -drdV);
        if (vtxData.position.y > maxHeight) maxHeight = vtxData.position.y;
        if (vtxData.position.y < minHeight) minHeight = vtxData.position.y;
        return vtxData;
    }

    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;    // vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float) j / M, (float) i / N));
                vtxData.push_back(GenVertexData((float) j / M, (float) (i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = HEIGHT
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *) offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *) offsetof(VertexData, normal));
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *) offsetof(VertexData, position.y));
    }

    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
    }
};


class Surface : public ParamSurface {
    int n = 6;
    float fi[12][12] = {};
    float A0 = 0.2f;
public:
    Surface() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fi[i][j] = (float) rand() / (float) RAND_MAX * 2 * M_PI;
            }
        }
        create();
    }

    void eval(Dnum2 &U, Dnum2 &V, Dnum2 &X, Dnum2 &Y, Dnum2 &Z) {
        X = (U - 0.5f) * M_PI * 2;
        Z = (V - 0.5f) * M_PI * 2;
        Y = -2;


        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float Ai;
                if (i + j > 0) {
                    Ai = A0 / sqrtf(float((i * i)) + float((j * j)));
                    Y = Y + Cos(X * i + Z * j + fi[i][j]) * Ai;
                }
            }
        }
    }
};


//---------------------------
struct Object {
//---------------------------
    Shader *shader;
    Material *material;
    Geometry *geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
public:
    Object(Shader *_shader, Material *_material, Geometry *_geometry) :
            scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        material = _material;
        geometry = _geometry;
    }

    virtual void SetModelingTransform(mat4 &M, mat4 &Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) *
               ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }

    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        shader->Bind(state);
        geometry->Draw();
    }

    virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

class Cuboid : public Geometry {
    struct VertexData {
        vec3 position;
        vec3 normal;
    };
public:
    std::vector<unsigned int> indices;
    std::vector<vec3> normals;
    std::vector<vec3> vertices;

    Cuboid(float width, float height, float depth) {
        // Calculate half extents
        float halfWidth = width / 2.0f;
        float halfHeight = height / 2.0f;
        float halfDepth = depth / 2.0f;

        // Vertices
        vertices = {
                vec3(-halfWidth, -halfHeight, halfDepth),  // Vertex 1
                vec3(-halfWidth, halfHeight, halfDepth),   // Vertex 2
                vec3(halfWidth, halfHeight, halfDepth),    // Vertex 3
                vec3(halfWidth, -halfHeight, halfDepth),   // Vertex 4

                vec3(-halfWidth, -halfHeight, -halfDepth),  // Vertex 5
                vec3(-halfWidth, halfHeight, -halfDepth),   // Vertex 6
                vec3(halfWidth, halfHeight, -halfDepth),    // Vertex 7
                vec3(halfWidth, -halfHeight, -halfDepth)    // Vertex 8
        };

        // Indices for rendering triangles
        indices = {
                // Front face
                0, 1, 2,
                2, 3, 0,

                // Top face
                1, 5, 6,
                6, 2, 1,

                // Back face
                7, 6, 5,
                5, 4, 7,

                // Bottom face
                4, 0, 3,
                3, 7, 4,

                // Left face
                4, 5, 1,
                1, 0, 4,

                // Right face
                3, 2, 6,
                6, 7, 3
        };

        create();
    }

    void create() {
        std::vector<VertexData> vtxData;
        for (int i = 0; i < indices.size() / 3; i++) {
            VertexData vtData;
            vec3 normal = normalize(cross(vertices[indices[i * 3 + 1]] - vertices[indices[i * 3]],
                                          vertices[indices[i * 3 + 2]] - vertices[indices[i * 3]]));
            std::vector<vec3> normals = {normal, normal, normal};
            vtData = {vertices[indices[i * 3]], normal};
            vtxData.push_back(vtData);
            vtData = {vertices[indices[i * 3 + 1]], normal};
            vtxData.push_back(vtData);
            vtData = {vertices[indices[i * 3 + 2]], normal};
            vtxData.push_back(vtData);
        }
        glBufferData(GL_ARRAY_BUFFER, indices.size() * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *) offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void *) offsetof(VertexData, normal));
    }


    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < indices.size(); i++) glDrawArrays(GL_TRIANGLE_STRIP, i * 3, 3);
    }
};

bool canStart = false;

//---------------------------
class Scene {
//---------------------------
    std::vector<Object *> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
    RenderState state;
    RenderState objectState;
    Camera objectCamera;
    Object *cuboidObject;
    const float a = 0.5f;
    const float b = 1.0f;
    const float c = 0.7f;
    const float m = 0.2f; //weight
    const float D = 0.8f; //rugoallando
    const float l0 = 1.6f; //kotel nyugalmi hossz
    const float I = m * (a * a + b * b) / 12; //tehetetlensegi nyomaték
    vec3 fi = vec3(0,0,0); //szogelfordulas
    const vec3 start = vec3(0, 2.5, 0); //forgas pontja a kotelnek
    const vec3 g = vec3(0, -9.81, 0); //gravitáció
    vec3 p = vec3(0.1f, 0, 0); //lendulet
    vec3 L = vec3(0, 0, 0); //perdulet

    vec3 center = vec3(0, 3, 0);

public:
    void Step(float Dt) {
        vec3 l = center - start;
        vec3 K;
        if (l0 < length(start - l)) {
            K = D * normalize(start-l) * (length(start - l) - l0);
        } else {
            K = 0;
        }
        vec3 v = p / m;
        vec3 F = m * g + K - p*v;
        p = p + F * Dt;
        center = center + v*Dt;

        vec3 omega = powf(I, -1) * L;
        vec3 M = cross((l-c), K) - 0.0001f * omega;
        fi = fi + powf(I, -1) * L * Dt;
        L = L + M * Dt;
        cuboidObject->rotationAngle = length(fi);
        cuboidObject->translation = center;
    }

    void Build() {
        // Shaders
        Shader *phongShader = new MapShader();
        Shader *jumperShader = new JumperShader();

        // Materials
        Material *material0 = new Material;
        material0->kd = vec3(0, 1, 0);
        material0->ks = vec3(0.5, 0.5, 0.5);


        Material *material1 = new Material;
        material1->kd = vec3(0.6, 0.4, 0.2);
        material1->ks = vec3(0.5, 0.5, 0.5);


        // Geometries
        Geometry *surface = new Surface();

        // Create objects by setting up their vertex data on the GPU

        Object *surfaceObject = new Object(phongShader, material0, surface);
        surfaceObject->translation = vec3(0, 1.0f, 0);
        surfaceObject->scale = vec3(1.7f, 1.7f, 1.7f);
        surfaceObject->shader = phongShader;
        objects.push_back(surfaceObject);
        Cuboid *cuboid = new Cuboid(a, b, c);

        cuboidObject = new Object(jumperShader, material1, cuboid);
        cuboidObject->translation = vec3(0, 3, 0);
        cuboidObject->scale = vec3(0.7f, 0.7f, 0.7f);
        cuboidObject->shader = jumperShader;
        objects.push_back(cuboidObject);

        // Camera
        std::vector<vec3> points = {cuboid->vertices[cuboid->indices[3]], cuboid->vertices[cuboid->indices[4]],
                                    cuboid->vertices[cuboid->indices[5]]};


        objectCamera.wEye = (points[0] + points[1]) / 2 + cuboidObject->translation;
        objectCamera.wLookat = normalize(
                cross(cuboid->vertices[1] - cuboid->vertices[0], cuboid->vertices[2] - cuboid->vertices[0]));
        objectCamera.wVup = vec3(0, 1, 0);

        // Camera
        camera.wEye = vec3(0, 0, 8);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(0, 1, 0);

        // Lights
        lights.resize(1);
        lights[0].wLightPos = vec4(5, 5, 1, 0);    // ideal point -> directional light source
        lights[0].La = vec3(0.4f, 0.4f, 0.4);
        lights[0].Le = vec3(0.4, 0.4, 0.4);
    }

    void Render() {
        mat4 M, Minv;
        glViewport(windowWidth / 2, 0, windowWidth / 2, windowHeight);
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object *obj: objects) obj->Draw(state);
        glViewport(0, 0, windowWidth / 2, windowHeight);
        cuboidObject->SetModelingTransform(M, Minv);
        Cuboid *cuboid = (Cuboid *) cuboidObject->geometry;
        std::vector<vec3> points = {cuboid->vertices[cuboid->indices[6]], cuboid->vertices[cuboid->indices[7]],
                                    cuboid->vertices[cuboid->indices[8]]};
        vec3 norm = cross(points[1] - points[0], points[2] - points[0]);
        vec4 tempLookat = vec4(norm.x, norm.y, norm.z, 1) * Minv;
        vec4 tempUp = vec4(points[0].x, points[0].y, points[0].z, 1) * Minv;

        for (int i = 0; i < points.size(); i++) {
            vec4 point = vec4(points[i].x, points[i].y, points[i].z, 1) * M;
            points[i] = vec3(point.x, point.y, point.z);
        }
        objectCamera.wEye = (points[0] + points[1]) / 2;
        objectCamera.wLookat = vec3(tempLookat.x, tempLookat.y, tempLookat.z);
        objectCamera.wVup = vec3(tempUp.x, tempUp.y, tempUp.z);
        objectState.wEye = objectCamera.wEye;
        objectState.V = objectCamera.V();
        objectState.P = objectCamera.P();
        objectState.lights = lights;
        for (Object *obj: objects) obj->Draw(objectState);
    }

    void Animate(float tstart, float tend) {
        vec4 newEye = vec4(camera.wEye.x, camera.wEye.y, camera.wEye.z, 1) * RotationMatrix(0.001f, vec3(0, 1, 0));
        camera.wEye = vec3(newEye.x, newEye.y, newEye.z);
        if (canStart) {
            Step((tend - tstart)/5.0f);
        }
    }
};

Scene scene;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    canStart = true;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is ”infinitesimal”
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
