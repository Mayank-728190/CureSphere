<?php
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_FILES["file"])) {
    $uploadDir = "uploads/";
    if (!file_exists($uploadDir)) {
        mkdir($uploadDir, 0777, true);
    }

    $fileName = basename($_FILES["file"]["name"]);
    $timestamp = date("Y-m-d H:i:s");
    $targetFile = $uploadDir . time() . "_" . $fileName;
    $fileLink = "uploads/" . basename($targetFile);

    if (move_uploaded_file($_FILES["file"]["tmp_name"], $targetFile)) {
        $csvData = [$fileName, $fileLink, $timestamp];
        $file = fopen("patient_records.csv", "a");
        fputcsv($file, $csvData);
        fclose($file);

        echo "File uploaded successfully!";
    } else {
        echo "Error uploading file.";
    }
} else {
    echo "No file uploaded.";
}
?>
