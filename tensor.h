#pragma once
#pragma once
#include <stdarg.h>
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <immintrin.h>


struct Vector {
	int size;
	int end;
	float *arr;
	__m256 first, second, third, fourth;

	Vector() {}

	Vector(int s) {
		size = s;
		end = (size / 8) * 8;
		arr = (float*)_aligned_malloc(sizeof(float) * s, 32);
	}

	~Vector() {
		free(arr);
	}

	void init(int s, float *a) {
		size = s;
		end = (s / 8) * 8;
		arr = a;
	}

	void zeros() {
		for (int i = end; i < size; i++) {
			arr[i] = 0;
		}
	}

	void randuniform(float low, float high) {
		for (int i = 0; i < size; i++) {
			arr[i] = (float)rand() / RAND_MAX;
			arr[i] = arr[i] * (high - low) + low;
		}
	}

	void print() {
		printf("[ ");
		for (int i = 0; i < size; i++) {
			printf("%f ", arr[i]);
		}
		printf("]\n");
	}

	void add(Vector &a, Vector &b) {
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(a.arr + i);
			second = _mm256_load_ps(b.arr + i);
			_mm256_store_ps(arr + i, _mm256_add_ps(first, second));
		}
		for (int i = end; i<size; i++) {
			arr[i] = a.arr[i] + b.arr[i];
		}
	}

	void entrymul(Vector &a, Vector &b) {
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(a.arr + i);
			second = _mm256_load_ps(b.arr + i);
			_mm256_store_ps(arr + i, _mm256_mul_ps(first, second));
		}
		for (int i = end; i<size; i++) {
			arr[i] = a.arr[i] * b.arr[i];
		}
	}

	float dot(Vector &a) {
		third = _mm256_setzero_ps();
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(arr + i);
			second = _mm256_load_ps(a.arr + i);
			third = _mm256_fmadd_ps(first, second, third);
		}
		fourth = _mm256_permute2f128_ps(third, third, 1);
		third = _mm256_hadd_ps(third, fourth);
		third = _mm256_hadd_ps(third, third);
		third = _mm256_hadd_ps(third, third);
		float k = _mm256_cvtss_f32(third);

		for (int i = a.end; i<a.size; i++) {
			k += arr[i] * a.arr[i];
		}

		return k;
	}

	void cmul(Vector &a, float k) {
		second = _mm256_set1_ps(k);
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(a.arr + i);
			_mm256_store_ps(arr + i, _mm256_mul_ps(first, second));
		}
		for (int i = end; i<size; i++) {
			arr[i] = a.arr[i] * k;
		}
	}

	void cmuladd(Vector &a, float k) {
		second = _mm256_set1_ps(k);
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(a.arr + i);
			third = _mm256_load_ps(arr + i);
			third = _mm256_fmadd_ps(first, second, third);
		}
		for (int i = end; i<size; i++) {
			arr[i] += a.arr[i] * k;
		}
	}

	float sum(float r=1.0f) {
		second = _mm256_setzero_ps();
		for (int i = 0; i<end; i += 8) {
			first = _mm256_load_ps(arr + i);
			second = _mm256_add_ps(first, second);
		}
		third = _mm256_permute2f128_ps(second, second, 1);
		second = _mm256_hadd_ps(second, third);
		second = _mm256_hadd_ps(second, second);
		second = _mm256_hadd_ps(second, second);
		float k = _mm256_cvtss_f32(second);
		for (int i = end; i<size; i++) {
			k += arr[i];
		}
		return k*r;
	}
};

struct Matrix {
	int row, column;
	Vector *cols;
	Matrix(int r, int c) {
		row = r;
		column = c;
		float *arr = (float*)_aligned_malloc(sizeof(float)*r*c, 32);
		cols = new Vector[c];
		for (int i = 0; i<c; i++) {
			cols[i].init(r, arr + i * r);
		}
	}

	void randuniform(float low, float high) {
		for (int i = 0; i < column; i++) {
			cols[i].randuniform(low, high);
		}
	}

	void print() {
		printf("\n");
		for (int i = 0; i < row; i++) {
			printf("[ ");
			for (int j = 0; j < column; j++) {
				printf("%f ", cols[j].arr[i]);
				
			}
			printf("]\n");
		}
		printf("\n");
	}

	void add(Matrix &a, Matrix &b) {
		for (int i = 0; i < column; i++) {
			cols[i].add(a.cols[i], b.cols[i]);
		}
	}

	void mul(Matrix &a, Matrix &b) {
		for (int i = 0; i < column; i++) {
			cols[i].zeros();
			for (int j = 0; j < a.column; j++) {
				cols[i].cmuladd(a.cols[j], b.cols[i].arr[j]);
			}
		}
	}

	void dot(Matrix &a, Matrix &b) {
		for (int i = 0; i < column; i++) {
			for (int j = 0; j < row; j++) {
				cols[i].arr[j] = a.cols[j].dot(b.cols[i]);
			}
		}
	}

	void outer(Vector &a, Vector &b) {
		for (int i = 0; i < column; i++) {
			cols[i].cmul(a, b.arr[i]);
		}
	}

};