#include <iostream>
#include <complex>

using namespace std;

int test() {
  {
    return 5;
  }
  {
    return 7;
  }
}

int main() {
  {
  struct Foo {
    int a;
    int b;
    __float128 x;
    std::complex<__float128> y;
  };

  Foo f;
  f.a = 1;
  f.b = 2;
  cout << f.a << " " << f.b << endl;
  }

  {
  struct Foo {
    float a;
    float b;
    enum Val {
      A = 4
    };
  };

  Foo f;
  f.a = 3.14159;
  f.b = 2.71828;
  cout << f.a << " " << f.b << " " << Foo::A << endl;
  }

  cout << test() << endl;
  return EXIT_SUCCESS;
}
