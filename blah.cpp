#include <boost/any.hpp>
#include <iostream>
#include <functional>

template<typename T>
bool compare(boost::any a, boost::any b) {
        return boost::any_cast<T>(a) == boost::any_cast<T>(b);
}

template<>
bool compare<std::string>(boost::any a, boost::any b) {
        return boost::any_cast<std::string>(a).compare(boost::any_cast<std::string>(b)) == 0;
}

bool p(std::string a, std::string b) {
	return a.compare(b) == 0;
}

int main(int argc, char* argv[]) {
	boost::any anything = 0;
	boost::any something = (std::string)"asdf";
	boost::any bleh = (std::string)"asdf";
	boost::any another_thing = 1;
	if(compare<std::string>(bleh, something)) {
		std::cout << "success\n";
	}

	std::function<bool(std::string, std::string)> func = p;
}

