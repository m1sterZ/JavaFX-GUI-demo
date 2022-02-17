package sample;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.layout.AnchorPane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.scene.control.TextArea;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class Controller implements Initializable {

    private Stage stage;
    private FileChooser fileChooser = new FileChooser();

    @FXML
    public AnchorPane root;
    @FXML
    public TextArea textArea;

    public Controller() { }

    private Stage getStage() {
        stage = (Stage) root.getScene().getWindow();
        return stage;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {

    }

    @FXML
    public void changeWindow() throws Exception {
        Stage current = getStage();
        current.close();
        Second second = new Second();
        second.showWindow();
    }

    @FXML
    public void openFile() {
        fileChooser.setTitle("Open File");
        File file = fileChooser.showOpenDialog(stage);
        if (null != file) {
            readText(file);
        }
    }

    private void readText(File file) {
        String text;

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            while (null != (text = reader.readLine())) {
                textArea.appendText(text + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void init(Stage primaryStage) {
        this.stage = primaryStage;
    }
}
