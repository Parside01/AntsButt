.PHONY: run-go-cli
build-run:
	mkdir cmake-build-release
	cd cmake-build-release && cmake .. -G Ninja
	cd ..
	cmake --build cmake-build-release --target neural_network -j 6
	go run third_party/go-cli/main.go
run:
	go run third_party/go-cli/main.go
run-exe:
	./cmd/main.exe

default: build-run

