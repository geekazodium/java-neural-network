plugins {
    id("java")
}

group = "com.geekazodium"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

val lwjglVersion = "3.3.2"
val lwjglNatives = "natives-windows"

dependencies {
    implementation("com.google.code.gson:gson:2.10.1")

    implementation(platform("org.lwjgl:lwjgl-bom:$lwjglVersion"))
    implementation("org.lwjgl:lwjgl:$lwjglVersion")
    implementation("org.lwjgl:lwjgl-opencl:$lwjglVersion")
    implementation("org.lwjgl", "lwjgl-stb")
    runtimeOnly("org.lwjgl", "lwjgl", classifier = lwjglNatives)
    runtimeOnly("org.lwjgl", "lwjgl-stb", classifier = lwjglNatives)
}

tasks.test {
    useJUnitPlatform()
}


repositories {
    mavenCentral()
}