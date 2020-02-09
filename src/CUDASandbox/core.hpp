#pragma once

#ifdef CSB_PLATFORM_WINDOWS
	#ifdef CSB_BUILD_DLL
		#define CSB_API __declspec(dllexport)
	#else
		#define CSB_API __declspec(dllimport)
	#endif
#elif CSB_PLATFORM_UNIX
	#ifdef CSB_BUILD_DLL
	#define CSB_API __attribute__((visibility("default")))
	#else
	#define CSB_API
	#endif
#else
	#define CSB_API
#endif