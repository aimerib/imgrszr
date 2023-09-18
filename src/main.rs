extern crate rustface;
extern crate rayon;
extern crate indicatif;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use clap::Parser;
use image::{GenericImageView, imageops, ImageFormat};
use rustface::ImageData;
use std::fs;
use std::path::{Path, PathBuf};
use eyre::{eyre, Result, WrapErr};
use tracing::{error, warn};
use std::io::Cursor;

const MODEL_DATA: &[u8] = include_bytes!("model/seeta_fd_frontal_v1.0.bin");
const MIN_FACE_SIZE: u32 = 20;
const SCORE_THRESH: f64 = 2.0;
const PYRAMID_SCALE_FACTOR: f32 = 0.8;
const SLIDE_WINDOW_STEP_X: u32 = 4;
const SLIDE_WINDOW_STEP_Y: u32 = 4;

#[derive(Parser)]
struct Cli {
    /// The path to the image or folder to be resized.
    img_path: PathBuf,
    /// Resize dimensions. Format: widthxheight (e.g. 800x600)
    #[clap(short, long, default_value = "2000x2000")]
    size: String,
    /// Desired output format (png, jpg, gif, bmp, tiff)
    #[clap(short = 'f', long = "format", default_value = "jpg")]
    image_format: String,
    /// The path to save the resized image or folder for multiple images.
    #[clap(short, long)]
    output_path: Option<PathBuf>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {:?}", err);
    }
}

fn run() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
    .without_time()
    .with_max_level(tracing::Level::TRACE)
    .init();

    let args = Cli::parse();

    if !args.img_path.exists() {
        return Err(eyre!("The provided path does not exist: {}", args.img_path.display()));
    }

    if args.img_path.is_dir() {
        process_directory(&args)
    } else {
        Err(eyre!("Provided path is not a directory."))
    }
}

fn process_directory(args: &Cli) -> Result<()> {
    let entries: Vec<_> = fs::read_dir(&args.img_path)
        .wrap_err_with(|| format!("Failed to read directory: {}", args.img_path.display()))?
        .collect();

    // Create a new progress bar instance
    let pb = ProgressBar::new(entries.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
        .progress_chars("#>-"));

    entries.par_iter()
        .filter_map(|entry_result| {
            match entry_result {
                Ok(entry) => Some(entry),
                Err(e) => {
                    error!("Failed to read directory entry: {}", e);
                    None
                }
            }
        })
        .for_each(|entry| {
            let entry_path = entry.path();
            if image::open(&entry_path).is_ok() {
                if let Err(e) = process_image(&entry_path, &args.size, &args.image_format, args.output_path.as_ref()) {
                    error!("Failed processing image {}: {}", entry_path.display(), e);
                }
            } else {
                warn!("Skipping unsupported or broken file: {}", entry_path.display());
            }
            pb.inc(1);  // Increment progress bar after processing each image
        });

    pb.finish_with_message("All images processed!"); // Finish the progress bar with a message
    Ok(())
}

fn process_image(img_path: &Path, size: &str, image_format: &str, output_dir: Option<&PathBuf>) -> Result<()> {
    let dimensions: Vec<&str> = size.split('x').collect();
    if dimensions.len() != 2 {
        return Err(eyre!("Invalid size format. Expected format: widthxheight"));
    }
    let width: u32 = dimensions[0].parse()?;
    let height: u32 = dimensions[1].parse()?;

    let img = image::open(img_path)
        .wrap_err_with(|| format!("Failed to open image: {}", img_path.display()))?;

    let square_crop = face_gravity_crop(&img)?;
    let resized = imageops::resize(&square_crop, width, height, imageops::FilterType::Lanczos3);

    let output_format = determine_image_format(image_format)?;
    let output_path = determine_output_path(img_path, image_format, output_dir)?;

    // Create the directory if it doesn't exist
    if let Some(parent_dir) = output_path.parent() {
        if !parent_dir.exists() {
            fs::create_dir_all(parent_dir).wrap_err_with(|| format!("Failed to create directory: {}", parent_dir.display()))?;
        }
    }

    resized.save_with_format(output_path.clone(), output_format)
        .wrap_err_with(|| format!("Failed to save resized image: {}", output_path.display()))?;

    Ok(())
}

fn determine_image_format(image_format: &str) -> Result<ImageFormat> {
    match image_format.to_lowercase().as_str() {
        "png" => Ok(ImageFormat::Png),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        "gif" => Ok(ImageFormat::Gif),
        "bmp" => Ok(ImageFormat::Bmp),
        "tiff" => Ok(ImageFormat::Tiff),
        _ => Err(eyre!("Unsupported format: {}", image_format))
    }
}

fn face_gravity_crop(img: &image::DynamicImage) -> Result<image::DynamicImage> {
    let (width, height) = img.dimensions();
    let gray_img = img.to_luma8();
    let bytes = gray_img.into_raw();
    let image = ImageData::new(&bytes, width, height);

    let model_instance = rustface::read_model(Cursor::new(MODEL_DATA))
        .wrap_err("Failed to read the model from bytes")?;

    let mut detector = rustface::create_detector_with_model(model_instance);

    detector.set_min_face_size(MIN_FACE_SIZE);
    detector.set_score_thresh(SCORE_THRESH);
    detector.set_pyramid_scale_factor(PYRAMID_SCALE_FACTOR);
    detector.set_slide_window_step(SLIDE_WINDOW_STEP_X, SLIDE_WINDOW_STEP_Y);

    if let Some(face) = detector.detect(&image).into_iter().next() {
        let dimension = width.min(height);
        let face_center_x = face.bbox().x() + (face.bbox().width() / 2) as i32;
        let face_center_y = face.bbox().y() + (face.bbox().height() / 2) as i32;

        let x = (face_center_x as u32).saturating_sub(dimension / 2);
        let y = (face_center_y as u32).saturating_sub(dimension / 2);

        Ok(img.crop_imm(x, y, dimension, dimension))
    } else {
        Ok(center_crop(img))
    }
}

fn center_crop(img: &image::DynamicImage) -> image::DynamicImage {
    let (width, height) = img.dimensions();
    let dimension = width.min(height);
    let x = (width / 2) - (dimension / 2);
    let y = (height / 2) - (dimension / 2);
    img.crop_imm(x, y, dimension, dimension)
}

fn determine_output_path(original_path: &Path, format: &str, output_dir: Option<&PathBuf>) -> Result<PathBuf> {
    let file_stem = original_path.file_stem()
        .ok_or_else(|| eyre!("Failed to get the file stem for: {}", original_path.display()))?;

    let mut new_filename = file_stem.to_string_lossy().to_string();
    new_filename += "_resized.";
    // new_filename += format;

    let extension = if format == "jpeg" { "jpeg" } else { format };

    new_filename += &format!("_resized.{}", extension);

    Ok(if let Some(dir) = output_dir {
        dir.join(new_filename)
    } else {
        original_path.parent().unwrap_or_else(|| Path::new(".")).join(new_filename)
    })
}

