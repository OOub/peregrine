solution 'gmm'
    configurations {'Release', 'Debug'}
    location 'build'

    for index, file in pairs(os.matchfiles('source/**.cpp')) do
    	local name = path.getbasename(file)
    	project(name)
    		-- General settings
    		kind 'ConsoleApp'
    		language 'C++'
        	location 'build'

			-- All files in source
        	files {'source/**.hpp',
                    file
        	}

	        -- Declare the configurations
	        configuration 'Release'
	            targetdir 'build/release'
	            defines {'NDEBUG','STATS_ENABLE_BLAZE_WRAPPERS'}
	            flags {'OptimizeSpeed'}

	        configuration 'Debug'
	            targetdir 'build/debug'
	            defines {'DEBUG','STATS_ENABLE_BLAZE_WRAPPERS'}
	            flags {'Symbols'}

	        configuration 'linux or macosx'
            	includedirs {'/usr/local/include'}
	        	libdirs {'/usr/local/lib'}
  				linkoptions {'-ltbb','-lopenblas'}

	        -- Linux specific settings
	        configuration 'linux'
                links {'pthread'}
	            buildoptions {'-std=c++14'}
	            linkoptions {'-std=c++14'}

	        -- Mac OS X specific settings
	        configuration 'macosx'
	            buildoptions {'-std=c++14'}
	            linkoptions {'-std=c++14'}
end
