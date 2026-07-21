use google_cloud_storage::client::Storage;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read bucket name and object name from environment or use a default
    let bucket_name = env::var("BUCKET_NAME").unwrap_or_else(|_| "my-bucket".to_string());
    let object_name = env::var("OBJECT_NAME").unwrap_or_else(|_| "my-object".to_string());

    // Inject global custom headers that apply to all requests
    let client = Storage::builder()
        .with_custom_header("x-custom-global-header", "my-global-value")
        .with_custom_header("x-goog-custom-project", "my-project")
        .build()
        .await?;

    println!(
        "Reading object '{}/{}' with global custom headers...",
        bucket_name, object_name
    );

    // This request will automatically include the global custom headers
    let result = client
        .read_object(format!("projects/_/buckets/{}", bucket_name), object_name)
        .send()
        .await;

    match result {
        Ok(_) => println!("Successfully read object!"),
        Err(e) => println!("Error reading object: {:?}", e),
    }

    Ok(())
}
