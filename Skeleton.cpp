/* Feladat:
 * A Szíriusz csillagképből érkező földönkívülieknek megtetszett a Word, Powerpoint, stb. 2D-s rajzoló programja, de azt közvetlenül nem használhatják,
 * ugyanis ők más geometriával dolgoznak. Önt bízták meg a program adaptálásával arra az egyszerű esetre, amikor három pontra egy háromszöget kell illeszteni.
 * Szerencsére adtak egy szótárat, ami a szíriuszi geometriai fogalmakat megfelelteti az általunk használt euklideszi geometriai fogalmaknak: A Szíriusz sík
 * az euklideszi sík egység sugarú köre, amit alapkörnek neveznek.
 * Miközben az euklideszi síkot a komplex számokkal lehet számszerűsíteni, a Szíriusz síkot az egységnél kisebb abszolút értékű komplex számokkal.
 * Amíg az euklideszi sík metrikája |dz|=sqrt(dx^2+dy^2) a Szíriusz síké |dz|/(1-|z|^2). A Szíriusz egyenes egy olyan euklideszi körív, amely az alapkörre merőlegesen érkezik.
 * A feladat három egér klikk után a három pont által definiált háromszöget a háttértől eltérő színnel kitölteni, az éleket ezektől eltérő színnel felrajzolni,
 * és a szabványos kimenetre a három szögeit és oldalainak hosszát kiírni.
 */

// This program has O(1) memory leak

#include "framework.h"

float degrees(float radians)
{
	return radians * 180.0f / M_PI;
}

// responsible for the identity circle
class Circle
{
public:
	Circle(const vec2& center, float radius, unsigned int numOfSides) : m_VAO(0)
	{
		unsigned int numOfVertices = numOfSides + 2;
		allVertices.reserve(numOfVertices);
		
		// center is on the 0. index
		allVertices.emplace_back(center);
		
		for (int i = 1; i < numOfVertices; ++i)
		{
			allVertices.emplace_back(
				// circle's equation
			center.x + (radius * cos(i * 2.0f * M_PI / numOfSides)),
			center.y + (radius * sin(i * 2.0f * M_PI / numOfSides))
			);
		}
		
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
	std::vector<vec2> allVertices; // stores the points on the circle
	unsigned int m_VAO;
};

class Curve
{
public:
	// p1,p2 are one of the 3 clicks
	static Curve Create(const vec2& p1, const vec2& p2)
	{
		vec2 center;
		
		/* calculating the center point
		 * solve:
		 * r^2 + 1 = |c|^2
		 * |p1 - c| = r -> |p1 - c|^2 = r^2
		 * Ax + By = Ax0 + By0
		 * where c(x, y) = ?; n(A, B) = p1 - p2; p0(x0, y0) = (p1 + p2) / 2
		 */
		float denominator = (2 * p2.x * p1.y - 2 * p1.x * p2.y); // both center.x and center.y has the same denominator
		center.x = (p1.y + p2.x * p2.x * p1.y + p1.y * p2.y * p2.y - p1.x * p1.x * p2.y - p1.y * p1.y * p2.y - p2.y) / denominator;
		center.y = (p1.x * p1.x * p2.x + p2.x * p1.y * p1.y - p1.x + p2.x - p1.x * p2.x * p2.x - p1.x * p2.y * p2.y) / denominator;
		
		// calculation the radius
		float radius = length(p1 - center);
		
		return Curve{ p1, p2, center, radius };
	}
	
	// calculate the 3 angles of the sirius triangle
	// curves: the 3 curve in order -> the 3 sides if the sirius triangle; points: the 3 clicks in order
	static std::vector<float> CalculateAngles(const std::vector<Curve>& curves, const std::vector<vec2>& points)
	{
		// represent a line with a normal vector and a p0 point
		struct Line
		{
			vec2 n;
			vec2 p0;
		};
		
		// calculate angle between two 2D vectors
		auto Angle = [](const vec2& v1, const vec2& v2){
			return acos(dot(v1, v2) / (length(v1) * length(v2)));
		};
		// does l1 and l2 have an intersection point? If yes return the point
		auto IntersectionPoint = [](const Line& l1, const Line& l2){
			vec2 point;
			// calculating intersection point
			float denominator = (l1.n.x * l2.n.y - l1.n.y * l2.n.x); // both point.x and point.y has the same denominator
			point.x = (l1.n.x * l2.n.y * l1.p0.x + l1.n.y * l2.n.y * l1.p0.y - l1.n.y * l2.n.x * l2.p0.x - l1.n.y * l2.n.y * l2.p0.y) / denominator;
			point.y = (l1.n.x * l2.n.x * l2.p0.x + l1.n.x * l2.n.y * l2.p0.y - l1.n.x * l2.n.x * l1.p0.x - l1.n.y * l2.n.x * l1.p0.y) / denominator;
			return point;
		};
		
		// how many sides do we have? (we always have 3)
		unsigned int count = curves.size();
		
		// helper for indexing
		auto at = [=](int idx){
			return (idx + count) % count;
		};
		
		std::vector<float> angles;
		for (int i = 0; i < count; ++i)
		{
			vec2 point;
			
			// how it works: https://www.geogebra.org/m/ctp2nkfu
			vec2 intersectionPoint1 = IntersectionPoint(
					Line{curves[at(i - 1)].center - points[at(i - 1)], points[at(i - 1)]},
			        Line{curves[at(i - 1)].center - points[at(i)], points[at(i)]}
					);
			
			vec2 intersectionPoint2 = IntersectionPoint(
					Line{curves[at(i)].center - points[at(i)], points[at(i)]},
					Line{curves[at(i)].center - points[at(i + 1)], points[at(i + 1)]}
					);
			
			vec2 v1 = intersectionPoint1 - points[i];
			vec2 v2 = intersectionPoint2 - points[i];
			
			angles.push_back(Angle(v1, v2));
		}
		return angles;
	}
	
public:
	// we can create a curve with 2 points, a center point and a radius. The curve is between the p1 and p2 (we always need the smaller curve)
	Curve(const vec2& p1, const vec2& p2, const vec2& c, float r) : center(c), radius(r)
	{
		// angle of p1 and p2. atan2 has an interval of (-pi, pi)
		fi1 = atan2(p1.y - center.y, p1.x - c.x);
		fi2 = atan2(p2.y - center.y, p2.x - c.x);
		
		DetermineSmallerAngle();
	}
	
	void AddPoints(std::vector<vec2>& vertices, float step)
	{
		// determining how we have to add the points
		if (fi1 > fi2)
			AddPointsClockwise(vertices, step);
		else
			AddPointsCounterClockwise(vertices, step);
	}
	
private:
	float fi1;
	float fi2;
	vec2 center;
	float radius;
	
private:
	void AddPointsClockwise(std::vector<vec2>& vertices, float step)
	{
		float t = fi1;
		while (t > fi2)
		{
			vertices.emplace_back(center.x + (radius * cos(t)), center.y + (radius * sin(t)));
			t -= step;
		}
	}
	
	void AddPointsCounterClockwise(std::vector<vec2>& vertices, float step)
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
		// making sure we have 2 positive angle
		fi1 += 2*M_PI;
		fi2 += 2*M_PI;
		// if the difference of fi1 and fi2 is bigger than M_PI, then we have the bigger curve right now (we need the smaller curve)
		// if we add 2*pi to the smaller angle we get the smaller curve
		if (fi1 - fi2 > M_PI)
			fi2 += 2*M_PI;
		if (fi2 - fi1 > M_PI)
			fi1 += 2*M_PI;
	}
};

// line1(p11, 12), line2(p21, p22)
static bool IsCrossingLines(const vec2& p11, const vec2& p12, const vec2& p21, const vec2& p22)
{
	float denominator = (p11.x - p12.x)*(p21.y - p22.y) - (p21.x - p22.x)*(p11.y - p12.y);
	float t1 = ( (p21.x - p22.x)*(p12.y - p22.y) + (p22.x - p12.x)*(p21.y - p22.y) ) / denominator;
	float t2 = ( (p22.x - p12.x)*(p11.y - p12.y) + (p12.y - p22.y)*(p11.x - p12.x) ) / denominator;
	// if 0 <= t1 <= 1 and 0 <= t2 <= 1 then we have an intersection point
	return t1 > 0.00000001 && t1 < 0.99999999f && t2 > 0.00000001 && t2 < 0.99999999f;
}

class SiriusTriangle
{
public:
	// we can create a sirius triangle with 3 sides (curves) and the 3 clicks on the screen, which means the 3 point of the triangle
	// step controls the amount of points on the triangle
	SiriusTriangle(std::vector<Curve>& curves, float step, const std::vector<vec2>& points) : m_VAO(0), m_FillEBO(0)
	{
		// stroses where the side points start
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
		
		std::vector<float> angles = Curve::CalculateAngles(curves, points);
		printf("Alpha: %f Beta: %f Gamma: %f Angle sum: %f\n\n",
				degrees(angles[0]), degrees(angles[1]), degrees(angles[2]), degrees(angles[0] + angles[1] + angles[2]));
		
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
	
	std::vector<vec2> allVertices; // stores the points of the polygon
	std::vector<Triangle> indicesToDraw; // stores the indexes of the fill
	unsigned int m_VAO;
	unsigned int m_FillEBO;
	
private:
	
	// let us access vertices via indices
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
				// we found an ear, adding indices to the vector<>
				indices.push_back(Triangle{polygonVertices.GetIndex(i - 1), polygonVertices.GetIndex(i), polygonVertices.GetIndex(i + 1)});
				// removing the ear vertex
				polygonVertices.erase(polygonVertices.begin() + i);
				// start from the beginning
				i = 0;
				continue;
			}
			++i;
		}
		// adding the last triangle to the indices
		indices.push_back(Triangle{polygonVertices.GetIndex(0), polygonVertices.GetIndex(1), polygonVertices.GetIndex(2)});
		return indices;
	}
	
	static bool IsEar(const VerticesToIndicesConverter& vertices, int n)
	{
		// ear, if the diagonal is inside the polygon and is not cross any other side
		return IsInside(vertices, n) && !IsEarCrossingAnotherSide(vertices, n);
	}
	
	static bool IsEarCrossingAnotherSide(const VerticesToIndicesConverter& vertices, int n)
	{
		int verticesSize = vertices.size();
		// prev and next are the indices of the 2 endpoints of the diagonal
		int prev = (verticesSize + (n - 1)) % verticesSize;
		int next = (n + 1) % verticesSize;
		
		const vec2& p11 = vertices[prev];
		const vec2& p12 = vertices[next];
		for (int i = 0; i < verticesSize; ++i)
		{
			// skipping the vertices of other sides of the ear
			if (n == i || n == i - 1 || prev == i || prev == i - 1 || next == i || next == i - 1)
				continue;
			
			// a side of the polygon: (p21, p22)
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
		// testline form a point on the diagonal to infinity (untill 1.1)
		vec2 testLine_p1 = (vertices[n - 1] + vertices[n + 1]) / 2;
		vec2 testLine_p2 = vec2(1.1f, testLine_p1.y);
		for (int i = 0; i < vertices.size(); ++i)
		{
			if (IsCrossingLines(testLine_p1, testLine_p2, vertices[i - 1], vertices[i]))
				++crossCount;
		}
		// if crosscount is odd then the diagonal was inside
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
	
	// creating identity circle
	identityCircle = new Circle(vec2(0.0f, 0.0f), 1.0f, 50);
	
	// creating shader
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
			curves.push_back(Curve::Create(clicks[0], clicks[1])); // curve between p1 and p2
			curves.push_back(Curve::Create(clicks[1], clicks[2])); // curve between p2 and p3
			curves.push_back(Curve::Create(clicks[2], clicks[0])); // curve between p3 and p1
			
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
