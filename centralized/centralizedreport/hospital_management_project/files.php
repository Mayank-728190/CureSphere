<?php
$filePath = "patient_records.csv";
$files = [];

if (file_exists($filePath)) {
    $fileHandle = fopen($filePath, "r");
    while (($row = fgetcsv($fileHandle)) !== false) {
        $files[] = ["name" => $row[0], "path" => $row[1], "timestamp" => $row[2]];
    }
    fclose($fileHandle);
}

header('Content-Type: application/json');
echo json_encode($files);
?>
