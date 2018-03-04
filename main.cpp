#include "tensor.h"

void main() {
	/*Matrix a(2,3);
	a.randuniform(-1.0, 1.0);
	a.print();

	Matrix b(2,4);
	b.randuniform(-1.0, 1.0);
	b.print();

	Matrix c(3,4);
	c.dot(a, b);
	c.print();
	*/
	//printf("%f\n%f", a.dot(b), a.sum(1.5));

	Vector a(3);
	a.randuniform(-1.0, 1.0);
	a.print();

	Vector b(3);
	b.randuniform(-1.0, 1.0);
	b.print();

	Matrix c(3, 3);
	c.outer(a, b);
	c.print();

	getchar();
}