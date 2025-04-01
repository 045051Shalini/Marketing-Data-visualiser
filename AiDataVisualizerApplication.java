import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;

@SpringBootApplication
public class AiDataVisualizerApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiDataVisualizerApplication.class, args);
    }
}

@RestController
@RequestMapping("/api")
class CodeExecutionController {

    @PostMapping("/execute")
    public String executeCode(@RequestBody CodeRequest request) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("python", "-c", request.getCode());
            Process process = processBuilder.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            catch (Exception e) {
            return "Execution Error: " + e.getMessage();
        }
    }
}

class CodeRequest {
    private String code;

    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }
}
