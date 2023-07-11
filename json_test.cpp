#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <string>
#include <json/json.h>

using namespace std;

string str;

int main()
{
	Json::Value root;
	root["id"] = "Luna";
	root["name"] = "Silver";
	root["age"] = 19;
	
	Json::Value items;
	items.append("nootbook");
	items.append("nootbook2");
	items.append("nootbook3");
	root["items"] = items;
	
	Json::StyledWriter writer;
	str = writer.write(root);
	cout << str << endl;

	cout << root["items"] << endl;
	
	return 0;
}