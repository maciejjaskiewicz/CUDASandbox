#pragma once

#ifdef CUDASB_PLATFORM_WINDOWS
	#ifdef CUDASB_BUILD_DLL
		#define CUDASB_API __declspec(dllexport)
	#else
		#define CUDASB_API __declspec(dllimport)
	#endif
#else
	#define CUDASB_API
#endif