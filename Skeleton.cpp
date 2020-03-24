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
		
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * numOfVertices * 2, allCircleVertices, GL_STATIC_DRAW);
		
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
		glEnableVertexAttribArray(0);
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

class Curve
{
public:
	static Curve Create(vec2 p1, vec2 p2)
	{
		float x1 = p1.x;
		float y1 = p1.y;
		float x2 = p2.x;
		float y2 = p2.y;
		vec2 c;
		c.x = (y1 + x2*x2*y1 + y1*y2*y2 - x1*x1*y2 - y1*y1*y2 - y2) / (2*x2*y1 - 2*x1*y2);
		c.y = (x1*x1*x2 + x2*y1*y1 - x1 + x2 - x1*x2*x2 - x1*y2*y2) / (2*x2*y1 - 2*x1*y2);
		float r = length(p1 - c);
		return Curve{p1, p2, c, r};
	}
	
	static float CalculateAngles(Curve& curve1, Curve& curve2, vec2 point)
	{
		vec2 v1 = curve1.center - point;
		vec2 v2 = curve2.center - point;
		float angle;
		angle = M_PI - acos(dot(v1, v2) / (length(v1) * length(v2)));
		return angle;
	}
	
public:
	Curve(const vec2& p1, const vec2& p2, const vec2& c, float r) : center(c), radius(r)
	{
		fi1 = atan2(p1.y - center.y, p1.x - c.x);
		fi2 = atan2(p2.y - center.y, p2.x - c.x);
		
		if (fi1 < 0 && fi2 > 0 && fi2 > M_PI / 2.0f)
		{
			fi1 += 2*M_PI;
		}
		else if (fi1 < 0 && fi2 > 0 && fi2 <= M_PI / 2.0f)
		{
			fi1 += 2*M_PI;
			fi2 += 2*M_PI;
		}
		if (fi2 < 0 && fi1 > 0 && fi1 > M_PI / 2.0f)
		{
			fi2 += 2*M_PI;
		}
		else if (fi2 < 0 && fi1 > 0 && fi1 <= M_PI / 2.0f)
		{
			fi1 += 2*M_PI;
			fi2 += 2*M_PI;
		}
	}
	
	void AddPoints(std::vector<vec2>& vertices, float step)
	{
		if (fi1 > fi2)
			AddPointsClockvise(vertices, step);
		else // if (c.fi1 < c.fi2)
			AddPointsAntiClockvise(vertices, step);
	}
	
	float Radius() const
	{
		return radius;
	}
	
private:
	float fi1;
	float fi2;
	vec2 center;
	float radius;
	
private:
	void AddPointsClockvise(std::vector<vec2>& vertices, float step)
	{
		float t = fi1;
		while (t > fi2)
		{
			vertices.emplace_back(center.x + (radius * cos(t)), center.y + (radius * sin(t)));
			t -= step;
		}
	}
	
	void AddPointsAntiClockvise(std::vector<vec2>& vertices, float step)
	{
		float t = fi1;
		while (t < fi2)
		{
			vertices.emplace_back(center.x + (radius * cos(t)), center.y + (radius * sin(t)));
			t += step;
		}
	}
};

bool operator==(const vec2& left, const vec2& right)
{
	return abs(left.x - right.x) < 0.00000001f && abs(left.y - right.y) < 0.00000001f;
}

bool CrossingLine(vec2 p11, vec2 p12, vec2 p21, vec2 p22)
{
	if (p11 == p21 || p11 == p22 || p12 == p21 || p12 == p22)
	{
		return false;
	}
	
	float x11 = p11.x;
	float y11 = p11.y;
	float x12 = p12.x;
	float y12 = p12.y;
	float x21 = p21.x;
	float y21 = p21.y;
	float x22 = p22.x;
	float y22 = p22.y;
	
	float t1 = ( (x21 - x22)*(y12 - y22) + (x22 - x12)*(y21 - y22) ) / ( (x11 - x12)*(y21 - y22) - (x21 - x22)*(y11 - y12) );
	float t2 = ( (x22 - x12)*(y11 - y12) + (y12 - y22)*(x11 - x12) ) / ( (x11 - x12)*(y21 - y22) - (x21 - x22)*(y11 - y12) );
	return t1 > 0 && t1 < 1.0f && t2 > 0 && t2 < 1.0f;
}

bool Inside(vec2 p11, vec2 p12, std::vector<unsigned int>& indexes, const std::vector<vec2>& vertices)
{
	for (unsigned int i = 0; i < indexes.size() - 1; ++i)
	{
		vec2 p21 = vertices[indexes[i]];
		vec2 p22 = vertices[indexes[i + 1]];
		if (CrossingLine(p11, p12, p21, p22))
		{
			return false;
		}
	}
	if (CrossingLine(p11, p12, vertices[indexes[0]], vertices[indexes[indexes.size() - 1]]))
	{
		return false;
	}
	else
	{
		vec2 p11test = (p11 + p12) / 2;
		vec2 p12test = vec2(1.1f, p11test.y);
		unsigned int crossCount = 0;
		for (unsigned int i = 0; i < indexes.size() - 1; ++i)
		{
			vec2 p21 = vertices[indexes[i]];
			vec2 p22 = vertices[indexes[i + 1]];
			if (CrossingLine(p11test, p12test, p21, p22))
			{
				++crossCount;
			}
		}
		if (CrossingLine(p11test, p12test, vertices[indexes[0]], vertices[indexes[indexes.size() - 1]]))
			++crossCount;
		
		return crossCount % 2 == 1; // paratlan
	}
}

struct Triangle
{
	unsigned int a;
	unsigned int b;
	unsigned int c;
};

class SiriusTriangle
{
public:
	SiriusTriangle(std::vector<Curve>& curves, float step, const std::vector<vec2>& points) : m_VAO(0)
	{
		std::vector<unsigned int> sideBeginIndex;
		for (auto& curve : curves)
		{
			float _step = step / curve.Radius();
			
			sideBeginIndex.push_back(allVertices.size());
			
			curve.AddPoints(allVertices, _step);
		}
		
		printf("Oldal: %f\n", CalculateTriangleSideLength(sideBeginIndex[0], sideBeginIndex[1]));
		printf("Oldal: %f\n", CalculateTriangleSideLength(sideBeginIndex[1], sideBeginIndex[2]));
		printf("Oldal: %f\n", CalculateTriangleSideLength(sideBeginIndex[2], allVertices.size()));
		
		float angle1 = degrees(Curve::CalculateAngles(curves[0], curves[1], points[1]));
		float angle2 = degrees(Curve::CalculateAngles(curves[1], curves[2], points[2]));
		float angle3 = degrees(Curve::CalculateAngles(curves[2], curves[0], points[0]));
		printf("Szog: %f\n", angle1);
		printf("Szog: %f\n", angle2);
		printf("Szog: %f\n", angle3);
		printf("Szogosszeg: %f\n", (angle1 + angle2 + angle3));
		
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * allVertices.size(), allVertices.data(), GL_STATIC_DRAW);
		
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
		glEnableVertexAttribArray(0);
		
		std::vector<unsigned int> indexes;
		indexes.reserve(allVertices.size());
		for (unsigned int i = 0; i < allVertices.size(); ++i)
			indexes.push_back(i);
		EarClipping(indexes, allVertices);
	}
	
	virtual ~SiriusTriangle()
	{
		glDeleteVertexArrays(1, &m_VAO);
		glDeleteBuffers(1, &m_FillEBO);
	}
	
	void DrawLines()
	{
		glBindVertexArray(m_VAO);
		glDrawArrays(GL_LINE_LOOP, 0, allVertices.size()); // GL_LINE_STRIP ??
	}
	
	void DrawFill()
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_FillEBO);
		glDrawElements(GL_TRIANGLES, indices.size() * 3, GL_UNSIGNED_INT, 0);
	}
	
private:
	std::vector<vec2> allVertices;
	std::vector<Triangle> indices;
	unsigned int m_VAO;
	unsigned int m_FillEBO;
	
private:
	
	void EarClipping(std::vector<unsigned int>& indexes, const std::vector<vec2>& vertices)
	{
		// the algorithm
		unsigned int i = 1;
		while (indexes.size() > 3)
		{
			vec2 p11 = vertices[indexes[(i - 1) % indexes.size()]];
			vec2 p12 = vertices[indexes[(i + 1) % indexes.size()]];
			if (Inside(p11, p12, indexes, vertices))
			{
				indices.emplace_back(Triangle{indexes[(i - 1) % indexes.size()], indexes[i % indexes.size()], indexes[(i + 1) % indexes.size()]});
				indexes.erase(indexes.begin() + (i % indexes.size()));
				continue;
			}
			++i;
		}
		indices.emplace_back(Triangle{indexes[0], indexes[1], indexes[2]});
		
		glGenBuffers(1, &m_FillEBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_FillEBO);
		
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle) * indices.size(), indices.data(), GL_STATIC_DRAW);
	}
	
	float CalculateTriangleSideLength(unsigned int begin, unsigned int end) const
	{
		float sideLength = 0;
		for (unsigned int i = begin; i < end; ++i)
		{
			vec2 v1;
			vec2 v2;
			if (i != allVertices.size() - 1)
			{
				v1 = allVertices[i + 1];
				v2 = allVertices[i];
			}
			else
			{
				v1 = allVertices[0];
				v2 = allVertices[i];
			}
			
			vec2 dz = v1 - v2;
			float ds = (sqrt(dz.x*dz.x + dz.y*dz.y)) / (1 - v1.x*v1.x - v1.y*v1.y);
			sideLength += ds;
		}
		return sideLength;
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
	glLineWidth(3);
	
	identityCircle = new Circle(0, 0, 1.0f, 50, false);
	identityCircleShaderProgram.create(circleVertexShader, circleFragmentShader, "outColor");
	identityCircleShaderProgram.Use();
	
	triangleShaderProgram.create(triangleVertexShader, triangleFragmentShader, "outColor");
	
	
	clicks.emplace_back(0.50f, -0.50f);
	clicks.emplace_back(-0.50f, -0.33f);
	clicks.emplace_back(-0.17f, 0.83f);
	std::vector<Curve> curves;
	curves.push_back(Curve::Create(clicks[0], clicks[1]));
	curves.push_back(Curve::Create(clicks[1], clicks[2]));
	curves.push_back(Curve::Create(clicks[2], clicks[0]));
	triangle = new SiriusTriangle(curves, 0.05f, clicks);
	clicks.clear();
	
	
	//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void onDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT);

	mat4 MVP = {
			1, 0, 0,0,
			0, 1, 0, 0,
			0, 0, 1,0,
			0, 0, 0,1
	};
	
	identityCircleShaderProgram.Use();
	identityCircleShaderProgram.setUniform(MVP, "MVP");
	identityCircleShaderProgram.setUniform(vec3(0.18f, 0.18f, 0.18f), "color");
	identityCircle->Draw();
	
	if (triangle != nullptr)
	{
		triangleShaderProgram.Use();
		triangleShaderProgram.setUniform(MVP, "MVP");
		triangleShaderProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
		triangle->DrawLines();
		triangleShaderProgram.setUniform(vec3(1.0f, 0.0f, 0.0f), "color");
		triangle->DrawFill();
	}
	
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY)
{

}

void onKeyboardUp(unsigned char key, int pX, int pY)
{
	if (key == 27) // escape
	{
		delete identityCircle;
		delete triangle;
		exit(0);
	}
}

void onMouseMotion(int pX, int pY)
{
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
}

void onMouse(int button, int state, int pX, int pY)
{
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
			std::vector<Curve> curves;
			curves.push_back(Curve::Create(clicks[0], clicks[1]));
			curves.push_back(Curve::Create(clicks[1], clicks[2]));
			curves.push_back(Curve::Create(clicks[2], clicks[0]));

			triangle = new SiriusTriangle(curves, 0.05f, clicks);
			
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
