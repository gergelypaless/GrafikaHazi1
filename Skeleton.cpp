//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
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
// Nev    : 
// Neptun : 
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

// --------------------------------------------------------------------
// Beadás elott kivenni:
// --------------------------------------------------------------------
#define ASSERT(x) if (!(x)) __builtin_trap() // csak GCC-vel tesztelve
#define GLCall(x) GLClearError(); x; ASSERT(GLLogCall(__FUNCTION__, __FILE__, __LINE__))

static void GLClearError()
{
	while (glGetError() != GL_NO_ERROR) ;
}
static bool GLLogCall(const char* function, const char* file, unsigned int line)
{
	if (unsigned int error = glGetError())
	{
		printf("[OpenGL Error] (%d): %s %s:%d\n", error, function, file, line);
		return false;
	}
	return true;
}
// --------------------------------------------------------------------


template <typename T>
struct SupportedVBOLayoutTypes {};
template <>
struct SupportedVBOLayoutTypes<float>
{
	static constexpr unsigned int glType = GL_FLOAT;
	static constexpr unsigned int normalized = GL_FALSE;
};

class LayoutElementBase
{
public:
	LayoutElementBase(unsigned int count, unsigned int type) : m_Count(count), m_Type(type) { }
	virtual ~LayoutElementBase() = default;
	unsigned int GetCount() const { return m_Count; }
	unsigned int GetType() const { return m_Type; }
	virtual unsigned int GetNormalized() const = 0;
	virtual unsigned int GetSize() const = 0;
private:
	unsigned int m_Count;
	unsigned int m_Type;
};

template<typename T>
struct LayoutElement : public LayoutElementBase
{
	explicit LayoutElement(unsigned int count) : LayoutElementBase(count, SupportedVBOLayoutTypes<T>::glType) {	}
	unsigned int GetSize() const override { return GetCount() * sizeof(T); }
	unsigned int GetNormalized() const override { return SupportedVBOLayoutTypes<T>::normalized; }
};

void UploadVertexBufferLayout(const std::vector<LayoutElementBase*>& layout)
{
	if (layout.empty())
		throw "empty vector of counts";
	
	unsigned int stride = 0;
	
	// determining stride value
	for (const auto& element : layout)
		stride += element->GetSize();
	
	unsigned long offset = 0;
	for (unsigned int i = 0; i < layout.size(); ++i)
	{
		glEnableVertexAttribArray(i);
		glVertexAttribPointer(i, layout[i]->GetCount(), layout[i]->GetType(), layout[i]->GetNormalized(), stride, (void*)offset);
		offset += layout[i]->GetSize();
	}
	
	for (const auto& element : layout)
		delete element;
}

float degrees(float radians)
{
	return radians * 180.0f / M_PI;
}
float radians(float degrees)
{
	return degrees * M_PI / 180.0f;
}

// TODO: refactor
class Circle
{
public:
	Circle(float x, float y, float radius, unsigned int numOfSides, bool isHollow = false) : m_VAO(0), isHollow(isHollow)
	{
		/* A kor pontjainak meghatarozasat ezek a forrasok segitsegevel hataroztam meg:
		 *      https://www.youtube.com/watch?v=YDCSKlFqpaU
		 *      https://www.youtube.com/watch?v=ccvebHuZOHM
		 */
		
		numOfVertices = numOfSides + 1 + (isHollow ? 0 : 1);
		
		float circleVerticesX[numOfVertices];
		float circleVerticesY[numOfVertices];
		
		if (!isHollow)
		{
			circleVerticesX[0] = x;
			circleVerticesY[0] = y;
		}
		
		for (int i = isHollow ? 0 : 1; i < numOfVertices; ++i)
		{
			circleVerticesX[i] = x + (radius * cos(i * 2.0f * M_PI / numOfSides));
			circleVerticesY[i] = y + (radius * sin(i * 2.0f * M_PI / numOfSides));
		}
		
		allCircleVertices = new float[numOfVertices * 2];
		
		for (int i = 0; i < numOfVertices; ++i)
		{
			allCircleVertices[i * 2 + 0] = circleVerticesX[i];
			allCircleVertices[i * 2 + 1] = circleVerticesY[i];
		}
		
		glGenVertexArrays(1, &m_VAO);	// get 1 vao id
		glBindVertexArray(m_VAO);		// make it active
		
		unsigned int vbo;		// vertex buffer object
		glGenBuffers(1, &vbo);	// Generate 1 buffer
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numOfVertices * 2, allCircleVertices, GL_STATIC_DRAW);
		
		UploadVertexBufferLayout({
			new LayoutElement<float>(2),
		});
	}
	
	virtual ~Circle()
	{
		delete[] allCircleVertices;
		glDeleteVertexArrays(1, &m_VAO);
	}
	
	void Draw()
	{
		glBindVertexArray(m_VAO);
		glDrawArrays(isHollow ? GL_LINE_LOOP : GL_TRIANGLE_FAN, 0, numOfVertices); // GL_LINE_STRIP ??
	}
	
private:
	float* allCircleVertices;
	
	unsigned int m_VAO;
	unsigned int numOfVertices;
	bool isHollow;
	
};

struct CurveData
{
	float fi1;
	float fi2;
	float centerX;
	float centerY;
	float radius;
};

class SiriusTriangle
{
public:
	SiriusTriangle(std::vector<CurveData>& curves, float step) : m_VAO(0)
	{
		for (auto& curve : curves)
		{
			float _step = step / curve.radius;
			
			ModifyAngles(curve);
			if (curve.fi1 > curve.fi2)
				AddPointsClockvise(curve, _step);
			else // if (c.fi1 < c.fi2)
				AddPointsAntiClockvise(curve, _step);
		}
		
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * allVertices.size(), allVertices.data(), GL_STATIC_DRAW);
		
		UploadVertexBufferLayout({
			new LayoutElement<float>(2)
		});
	}
	
	virtual ~SiriusTriangle()
	{
		glDeleteVertexArrays(1, &m_VAO);
	}
	
	void Draw()
	{
		glBindVertexArray(m_VAO);
		glDrawArrays(GL_LINE_LOOP, 0, allVertices.size()); // GL_LINE_STRIP ??
	}
	
private:
	std::vector<vec2> allVertices;
	unsigned int m_VAO;
	
private:
	void AddPointsClockvise(CurveData& curve, float step)
	{
		float t = curve.fi1;
		while (t > curve.fi2)
		{
			allVertices.emplace_back(curve.centerX + (curve.radius * cos(t)), curve.centerY + (curve.radius * sin(t)));
			t -= step;
		}
	}
	
	void AddPointsAntiClockvise(CurveData& curve, float step)
	{
		float t = curve.fi1;
		while (t < curve.fi2)
		{
			allVertices.emplace_back(curve.centerX + (curve.radius * cos(t)), curve.centerY + (curve.radius * sin(t)));
			t += step;
		}
	}
	
	static void ModifyAngles(CurveData& c)
	{
		if (c.fi1 < 0 && c.fi2 > 0 && c.fi2 > M_PI / 2.0f)
		{
			c.fi1 += 2*M_PI;
		}
		else if (c.fi1 < 0 && c.fi2 > 0 && c.fi2 <= M_PI / 2.0f)
		{
			c.fi1 += 2*M_PI;
			c.fi2 += 2*M_PI;
		}
		if (c.fi2 < 0 && c.fi1 > 0 && c.fi1 > M_PI / 2.0f)
		{
			c.fi2 += 2*M_PI;
		}
		else if (c.fi2 < 0 && c.fi1 > 0 && c.fi1 <= M_PI / 2.0f)
		{
			c.fi1 += 2*M_PI;
			c.fi2 += 2*M_PI;
		}
	}
};

const char * const circleVertexShader = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 aPos;	// Varying input: vp = vertex position is expected in attrib array 0

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix

	void main() {
		gl_Position = vec4(aPos.xy, 0.0f, 1.0f) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const circleFragmentShader = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	out vec4 outColor;		// computed color of the current pixel

	uniform vec3 color;		// uniform variable, the color of the primitive

	void main() {
		outColor = vec4(color.xyz, 1.0f);	// computed color is the color of the primitive
	}
)";

const char * const triangleVertexShader = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec2 aPos;	// Varying input: vp = vertex position is expected in attrib array 0

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix

	void main() {
		gl_Position = vec4(aPos.xy, 0.1f, 1.0f) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char * const triangleFragmentShader = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	out vec4 outColor;		// computed color of the current pixel

	uniform vec3 color;		// uniform variable, the color of the primitive

	void main() {
		outColor = vec4(color.xyz, 1.0f);	// computed color is the color of the primitive
	}
)";

GPUProgram identityCircleShaderProgram;
GPUProgram triangleShaderProgram;
Circle* identityCircle;

SiriusTriangle* triangle;
std::vector<vec2> clicks;

void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);
	
	//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	// identitty circle
	identityCircle = new Circle(0, 0, 1.0f, 50, false);
	identityCircleShaderProgram.create(circleVertexShader, circleFragmentShader, "outColor");
	identityCircleShaderProgram.Use();
	
	triangleShaderProgram.create(triangleVertexShader, triangleFragmentShader, "outColor");
	
	//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

// Window has become invalid: Redraw
void onDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	mat4 MVP = { 1, 0, 0, 0,    // MVP matrix,
		                      0, 1, 0, 0,    // row-major!
		                      0, 0, 1, 0,
		                      0, 0, 0, 1 };
	
	identityCircleShaderProgram.Use();
	identityCircleShaderProgram.setUniform(MVP, "MVP");
	identityCircleShaderProgram.setUniform(vec3(0.18f, 0.18f, 0.18f), "color");
	identityCircle->Draw();
	
	if (triangle != nullptr)
	{
		triangleShaderProgram.Use();
		triangleShaderProgram.setUniform(MVP, "MVP");
		triangleShaderProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
		triangle->Draw();
	}
	
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {

}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	if (key == 27) // escape
	{
		delete identityCircle;
		delete triangle;
		exit(0);
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

CurveData CreateCurve(vec2 p1, vec2 p2)
{
	float x1 = p1.x;
	float y1 = p1.y;
	float x2 = p2.x;
	float y2 = p2.y;
	vec2 c;
	c.x = (x1*x1 + y1*y1 + 1 - 2*y1*( (x1*x1*x2 + x2*y1*y1 - x1 + x2 - x1*x2*x2 - x1*y2*y2) / (2*x2*y1 - 2*x1*y2) )) / (2*x1);
	c.y = (x1*x1*x2 + x2*y1*y1 - x1 + x2 - x1*x2*x2 - x1*y2*y2) / (2*x2*y1 - 2*x1*y2);
	float r = length(p1 - c);
	return CurveData{atan2(y1 - c.y, x1 - c.x), atan2(y2 - c.y, x2 - c.x), c.x, c.y, r};
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	const char * buttonStat = nullptr;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
	
	// TODO: check for cX and cY are offside?
	if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
	{
		clicks.emplace_back(cX, cY);
		
		if (clicks.size() == 3)
		{
			std::vector<CurveData> curves;
			
			curves.push_back(CreateCurve(clicks[0], clicks[1]));
			curves.push_back(CreateCurve(clicks[1], clicks[2]));
			curves.push_back(CreateCurve(clicks[2], clicks[0]));

			triangle = new SiriusTriangle(curves, 0.01f);
			
			clicks.clear();
		}
		else
		{
			delete triangle;
			triangle = nullptr;
		}
	}
}



void onIdle()
{
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	glutPostRedisplay();
}
