package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func corsDisableMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

type ServiceIn struct {
	Pixels []float64 `json:"pixels"`
	Expect int       `json:"expect"`
}

type ServiceOut struct {
	Prediction int `json:"prediction"`
}

const (
	imageLearnDataFile      = "./learn_data/image.txt"
	neuralNetworkExecutable = "./cmake-build-release/neural_network.exe"
	networkWeightsFile      = "./network_weights.txt"
)

var (
	weightsAndBiases []float64
)

func readWeightsFromFile(filename string) ([]float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var weights []float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		nums := strings.Split(line, " ")
		for _, numStr := range nums {
			if numStr == "" {
				continue
			}
			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				log.Printf("Error parsing float from weights file: %v", err)
				return nil, err
			}
			weights = append(weights, num)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return weights, nil
}

func servicePredHandler(w http.ResponseWriter, r *http.Request) {
	var in ServiceIn
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, "Bad request (invalid JSON)", http.StatusBadRequest)
		return
	}

	if len(in.Pixels) != 784 {
		msg := fmt.Sprintf("Expected 784 pixels, but got %d", len(in.Pixels))
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

	var weightsStdin strings.Builder
	for _, value := range weightsAndBiases {
		weightsStdin.WriteString(fmt.Sprintf("%f\n", value))
	}

	var stdinData strings.Builder
	stdinData.WriteString(weightsStdin.String())
	for _, pixel := range in.Pixels {
		stdinData.WriteString(fmt.Sprintf("%f\n", pixel))
	}

	cmd := exec.Command(neuralNetworkExecutable)

	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		http.Error(w, "Internal server error (stdin pipe)", http.StatusInternalServerError)
		return
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		http.Error(w, "Internal server error (stdout pipe)", http.StatusInternalServerError)
		return
	}

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		http.Error(w, "Internal server error (stderr pipe)", http.StatusInternalServerError)
		return
	}

	if err := cmd.Start(); err != nil {
		http.Error(w, "Internal server error (start process)", http.StatusInternalServerError)
		return
	}

	go func() {
		defer stdinPipe.Close()
		_, err := io.WriteString(stdinPipe, stdinData.String())
		if err != nil {
			log.Printf("Error writing data to stdin pipe: %v", err)
		}
	}()

	var stdout, stderr bytes.Buffer
	_, errStdout := io.Copy(&stdout, stdoutPipe)
	_, errStderr := io.Copy(&stderr, stderrPipe)

	cmdErrWaiter := cmd.Wait()

	if cmdErrWaiter != nil {
		log.Printf("Neural network process finished with error: %v, stderr: %s", cmdErrWaiter, stderr.String())
		http.Error(w, fmt.Sprintf("Neural network execution failed: %s", stderr.String()), http.StatusInternalServerError)
		return
	}
	if errStdout != nil {
		log.Printf("Error reading stdout: %v", errStdout)
	}
	if errStderr != nil {
		log.Printf("Error reading stderr: %v", errStderr)
	}
	if stderr.Len() > 0 {
		log.Printf("Neural network process produced stderr output: %s", stderr.String())
		http.Error(w, fmt.Sprintf("Neural network stderr: %s", stderr.String()), http.StatusInternalServerError)
		return
	}

	predictionStr := strings.TrimSpace(stdout.String())
	prediction, err := strconv.Atoi(predictionStr)
	if err != nil {
		http.Error(w, "Internal server error (parsing prediction)", http.StatusInternalServerError)
		return
	}

	log.Printf("Prediction: %d", prediction)

	response := ServiceOut{Prediction: prediction}
	w.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w).Encode(response)
	if err != nil {
		log.Printf("Error encoding and writing response JSON: %v", err)
	}
}

func serviceLearnHandler(w http.ResponseWriter, r *http.Request) {
	var in ServiceIn
	if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
		http.Error(w, "Bad request (invalid JSON)", http.StatusBadRequest)
		return
	}

	if len(in.Pixels) != 784 {
		msg := fmt.Sprintf("Expected 784 pixels, but got %d", len(in.Pixels))
		http.Error(w, msg, http.StatusBadRequest)
		return
	}

	file, err := os.OpenFile(imageLearnDataFile, os.O_RDWR|os.O_CREATE, 0666)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error open file: %v", err), http.StatusInternalServerError)
		return
	}
	defer file.Close()

	for i := 0; i < 28; i++ {
		var lineBuilder strings.Builder
		for j := 0; j < 28; j++ {
			lineBuilder.WriteString(fmt.Sprintf("%d ", int32(in.Pixels[i*28+j])))
		}
		lineBuilder.WriteString("\n")
		if _, err = file.WriteString(lineBuilder.String()); err != nil {
			http.Error(w, fmt.Sprintf("Error writing line to file: %v", err), http.StatusInternalServerError)
			return
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
}

func main() {
	var err error
	if weightsAndBiases, err = readWeightsFromFile(networkWeightsFile); err != nil {
		panic(err)
	}
	http.HandleFunc("/pred", corsDisableMiddleware(servicePredHandler))
	http.HandleFunc("/learn", corsDisableMiddleware(serviceLearnHandler))
	if err := http.ListenAndServe(":8080", nil); err != nil {
		slog.Error("Error starting server: %s", err.Error())
	}
}
