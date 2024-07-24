<script setup lang="ts">
defineProps<{
  msg: string
}>()
</script>

<template>
  <div class="logo-container">
    <!-- <img alt="Vue logo" class="logo" src="@/assets/logo.png" width="360" /> -->
    <embed type="text/html" src="heatmap.html" width="80vw" height="800">
  </div>
</template>

<style scoped>
.logo-container {
  text-align: center;
}
/* h1 {
  font-weight: 500;
  font-size: 2.6rem;
  position: relative;
  top: -10px;
}

h3 {
  font-size: 1.2rem;
}

.greetings h1,
.greetings h3 {
  text-align: center;
}

@media (min-width: 1024px) {
  .greetings h1,
  .greetings h3 {
    text-align: left;
  }
} */
</style>
