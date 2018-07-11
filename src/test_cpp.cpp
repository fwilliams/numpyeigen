#include <iostream>

using namespace std;

int main() {
  {
  struct Foo {
    int a;
    int b;
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

  return EXIT_SUCCESS;
}
