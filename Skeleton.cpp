//=============================================================================================
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
// Nev    : Paless Gergely
// Neptun : Z7CHHP
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

class Circle
{
public:
	Circle(const vec2& center, float radius, unsigned int numOfSides) : m_VAO(0)
	{
		/* A kor pontjainak meghatarozasat ennek a videonak a segitsegevel hataroztam meg:
		 *      https://www.youtube.com/watch?v=YDCSKlFqpaU
		 */
		// innentol
		unsigned int numOfVertices = numOfSides + 2;
		allVertices.reserve(numOfVertices);
		allVertices.emplace_back(center);
		
		for (int i = 1; i < numOfVertices; ++i)
		{
			allVertices.emplace_back(
			center.x + (radius * cos(i * 2.0f * M_PI / numOfSides)),
			center.y + (radius * sin(i * 2.0f * M_PI / numOfSides))
			);
		}
		// idaig
		
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * allVertices.size(), allVertices.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
		glEnableVertexAttribArray(0);
	}
	
	virtual ~Circle()
	{
		glDeleteVertexArrays(1, &m_VAO);
	}
	
	void Draw()
	{
		glBindVertexArray(m_VAO);
		glDrawArrays(GL_TRIANGLE_FAN, 0, allVertices.size());
	}
	
private:
	std::vector<vec2> allVertices;
	unsigned int m_VAO;
};

class Curve
{
public:
	static Curve Create(const vec2& p1, const vec2& p2)
	{
		vec2 center;
		float denominator = (2 * p2.x * p1.y - 2 * p1.x * p2.y);
		center.x = (p1.y + p2.x * p2.x * p1.y + p1.y * p2.y * p2.y - p1.x * p1.x * p2.y - p1.y * p1.y * p2.y - p2.y) / denominator;
		center.y = (p1.x * p1.x * p2.x + p2.x * p1.y * p1.y - p1.x + p2.x - p1.x * p2.x * p2.x - p1.x * p2.y * p2.y) / denominator;
		float radius = length(p1 - center);
		return { p1, p2, center, radius };
	}
	
	static float CalculateAngles(Curve& curve1, Curve& curve2, const vec2& point)
	{
		vec2 v1 = curve1.center - point;
		vec2 v2 = curve2.center - point;
		float angle;
		angle = acos(dot(v1, v2) / (length(v1) * length(v2)));
		if (angle > M_PI / 2.0f)
			angle = M_PI - angle;
		
		return angle;
	}
	
public:
	Curve(const vec2& p1, const vec2& p2, const vec2& c, float r) : center(c), radius(r)
	{
		fi1 = atan2(p1.y - center.y, p1.x - c.x);
		fi2 = atan2(p2.y - center.y, p2.x - c.x);
		
		DetermineSmallerAngle();
	}
	
	void AddPoints(std::vector<vec2>& vertices, float step)
	{
		if (fi1 > fi2)
			AddPointsClockvise(vertices, step);
		else
			AddPointsAntiClockvise(vertices, step);
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
	
	void DetermineSmallerAngle()
	{
		fi1 += 2*M_PI;
		fi2 += 2*M_PI;
		if (fi1 - fi2 > M_PI)
			fi2 += 2*M_PI;
		if (fi2 - fi1 > M_PI)
			fi1 += 2*M_PI;
	}
};

static bool IsCrossingLines(const vec2& p11, const vec2& p12, const vec2& p21, const vec2& p22)
{
	float denominator = (p11.x - p12.x)*(p21.y - p22.y) - (p21.x - p22.x)*(p11.y - p12.y);
	float t1 = ( (p21.x - p22.x)*(p12.y - p22.y) + (p22.x - p12.x)*(p21.y - p22.y) ) / denominator;
	float t2 = ( (p22.x - p12.x)*(p11.y - p12.y) + (p12.y - p22.y)*(p11.x - p12.x) ) / denominator;
	return t1 > 0 && t1 < 1.0f && t2 > 0 && t2 < 1.0f;
}

class SiriusTriangle
{
public:
	SiriusTriangle(std::vector<Curve>& curves, float step, const std::vector<vec2>& points) : m_VAO(0), m_FillEBO(0)
	{
		std::vector<unsigned int> sideBeginIndex;
		sideBeginIndex.reserve(3);
		for (auto& curve : curves)
		{
			sideBeginIndex.push_back(allVertices.size());
			curve.AddPoints(allVertices, step);
		}
		
		float a = CalculateTriangleSideLength(sideBeginIndex[0], sideBeginIndex[1]);
		float b = CalculateTriangleSideLength(sideBeginIndex[1], sideBeginIndex[2]);
		float c = CalculateTriangleSideLength(sideBeginIndex[2], allVertices.size());
		printf("a: %f b: %f c: %f\n", a, b, c);
		
		float alpha = Curve::CalculateAngles(curves[0], curves[1], points[1]);
		float beta = Curve::CalculateAngles(curves[1], curves[2], points[2]);
		float gamma = Curve::CalculateAngles(curves[2], curves[0], points[0]);
		printf("Alpha: %f Beta: %f Gamma: %f Angle sum: %f\n\n", degrees(alpha), degrees(beta), degrees(gamma), degrees(alpha + beta + gamma));
		
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);
		
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * allVertices.size(), allVertices.data(), GL_STREAM_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, 0);
		glEnableVertexAttribArray(0);
		
		indicesToDraw = EarClipping(allVertices);
		glGenBuffers(1, &m_FillEBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_FillEBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Triangle) * indicesToDraw.size(), indicesToDraw.data(), GL_STREAM_DRAW);
	}
	
	virtual ~SiriusTriangle()
	{
		glDeleteVertexArrays(1, &m_VAO);
		glDeleteBuffers(1, &m_FillEBO);
	}
	
	void DrawLines()
	{
		glBindVertexArray(m_VAO);
		glDrawArrays(GL_LINE_LOOP, 0, allVertices.size());
	}
	
	void DrawFill()
	{
		glBindVertexArray(m_VAO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_FillEBO);
		glDrawElements(GL_TRIANGLES, indicesToDraw.size() * 3, GL_UNSIGNED_INT, 0);
	}
	
private:
	struct Triangle
	{
		unsigned int a;
		unsigned int b;
		unsigned int c;
	};
	
	std::vector<vec2> allVertices;
	std::vector<Triangle> indicesToDraw;
	unsigned int m_VAO;
	unsigned int m_FillEBO;
	
private:
	
	class VerticesToIndicesConverter : public std::vector<unsigned int>
	{
	public:
		explicit VerticesToIndicesConverter(const std::vector<vec2>& vertices) : std::vector<unsigned int>(), m_Vertices(vertices)
		{
			unsigned int verticesSize = vertices.size();
			reserve(verticesSize);
			for (unsigned int i = 0; i < verticesSize; i++)
				push_back(i);
		}
		
		const vec2& operator[](long idx) const
		{
			unsigned long siz = size();
			return m_Vertices[std::vector<unsigned int>::operator[]((idx + siz) % siz)];
		}
		
		unsigned int GetIndex(long idx) const
		{
			unsigned long siz = size();
			return std::vector<unsigned int>::operator[]((idx + siz) % siz);
		}
		
	private:
		const std::vector<vec2>& m_Vertices;
	};
	
	static std::vector<Triangle> EarClipping(const std::vector<vec2>& vertices)
	{
		std::vector<Triangle> indices;
		VerticesToIndicesConverter polygonVertices(vertices);
		
		int i = 0;
		while (polygonVertices.size() > 3 && i < polygonVertices.size())
		{
			if (IsEar(polygonVertices, i))
			{
				indices.push_back(Triangle{polygonVertices.GetIndex(i - 1), polygonVertices.GetIndex(i), polygonVertices.GetIndex(i + 1)});
				polygonVertices.erase(polygonVertices.begin() + i);
				i = 0;
			}
			++i;
		}
		indices.push_back(Triangle{polygonVertices.GetIndex(0), polygonVertices.GetIndex(1), polygonVertices.GetIndex(2)});
		return indices;
	}
	
	static bool IsEar(const VerticesToIndicesConverter& vertices, int n)
	{
		return IsInside(vertices, n) && !IsEarCrossingAnotherSide(vertices, n);
	}
	
	static bool IsEarCrossingAnotherSide(const VerticesToIndicesConverter& vertices, int n)
	{
		int verticesSize = vertices.size();
		int prev = (verticesSize + (n - 1)) % verticesSize;
		int next = (n + 1) % verticesSize;
		
		const vec2& p11 = vertices[prev];
		const vec2& p12 = vertices[next];
		for (int i = 0; i < verticesSize; ++i)
		{
			if (n == i || n == i - 1 || prev == i || prev == i - 1 || next == i || next == i - 1)
				continue;
			
			const vec2& p21 = vertices[i - 1];
			const vec2& p22 = vertices[i];
			if (IsCrossingLines(p11, p12, p21, p22))
				return true;
		}
		return false;
	}
	
	static bool IsInside(const VerticesToIndicesConverter& vertices, int n)
	{
		unsigned int crossCount = 0;
		vec2 testLine_p1 = (vertices[n - 1] + vertices[n + 1]) / 2;
		vec2 testLine_p2 = vec2(1.1f, testLine_p1.y);
		for (int i = 0; i < vertices.size(); ++i)
		{
			if (IsCrossingLines(testLine_p1, testLine_p2, vertices[i - 1], vertices[i]))
				++crossCount;
		}
		return crossCount % 2 == 1;
	}
	
	float CalculateTriangleSideLength(unsigned int begin, unsigned int end) const
	{
		unsigned int verticesSize = allVertices.size();
		
		float sideLength = 0;
		for (unsigned int i = begin; i < end; ++i)
		{
			vec2 v1;
			vec2 v2;
			if (i != verticesSize - 1)
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

const char * const vertexShaderSource = R"(
	#version 330
	precision highp float;

	layout(location = 0) in vec2 aPos;

	void main() {
		gl_Position = vec4(aPos.xy, 0.0f, 1.0f);
	}
)";

const char * const fragmentShaderSource = R"(
	#version 330
	precision highp float;
	
	out vec4 outColor;

	uniform vec3 color;

	void main() {
		outColor = vec4(color.xyz, 1.0f);
	}
)";

GPUProgram shaderProgram;
Circle* identityCircle;
SiriusTriangle* triangle;
std::vector<vec2> clicks;

void onInitialization()
{
	glViewport(0, 0, windowWidth, windowHeight);
	
	identityCircle = new Circle(vec2(0.0f, 0.0f), 1.0f, 50);
	shaderProgram.create(vertexShaderSource, fragmentShaderSource, "outColor");
	shaderProgram.Use();
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

void onDisplay()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	shaderProgram.Use();
	shaderProgram.setUniform(vec3(0.18f, 0.18f, 0.18f), "color");
	identityCircle->Draw();
	
	if (triangle != nullptr)
	{
		shaderProgram.setUniform(vec3(1.0f, 0.0f, 0.0f), "color");
		triangle->DrawFill();
		shaderProgram.setUniform(vec3(1.0f, 1.0f, 1.0f), "color");
		triangle->DrawLines();
	}
	
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) { }
void onKeyboardUp(unsigned char key, int pX, int pY) { }
void onMouseMotion(int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY)
{
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	
	if (state == GLUT_UP && button == GLUT_LEFT_BUTTON)
	{
		clicks.emplace_back(cX, cY);
		if (clicks.size() == 3)
		{
			std::vector<Curve> curves;
			curves.reserve(3);
			curves.push_back(Curve::Create(clicks[0], clicks[1]));
			curves.push_back(Curve::Create(clicks[1], clicks[2]));
			curves.push_back(Curve::Create(clicks[2], clicks[0]));

			triangle = new SiriusTriangle(curves, 0.008f, clicks);
			
			clicks.clear();
		}
		else
		{
			delete triangle;
			triangle = nullptr;
		}
		glutPostRedisplay();
	}
}

void onIdle() {}
